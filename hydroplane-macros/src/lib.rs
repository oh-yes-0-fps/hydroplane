//! The `#[kernel]` attribute — write a `hydroplane` kernel as a plain generic function.
//!
//! From a function whose first parameter is the SIMD context, `#[kernel]` generates the kernel
//! struct and `Kernel` impl (`MatrixKernel` with `#[kernel(matrix)]`, adding the `.tiles()`
//! surface), a callable wrapper that runs `dispatch` for you, and a `<name>_on(ctx, …)` companion
//! that runs the body on a caller-supplied context — call it from another kernel to skip dispatch.
//!
//! ```ignore
//! #[hydroplane::kernel]
//! fn any_overlap<'a, T: Scalar>(ctx: Gang, soa: &'a Soa<T>, q: [T; 4]) -> bool { /* body */ }
//! let hit = any_overlap(&soa, q);   // call site — no struct, no impl, no `dispatch`
//! ```
//!
//! The context is written bare (`ctx: Gang`) or concrete (`ctx: Gang<f32>`); every later parameter
//! becomes a struct field carried across the dispatch boundary, so borrows need explicit
//! lifetimes. The dispatch element is the generic bound by `Scalar`/`FloatScalar`/`IntScalar` (or
//! the one named `T`), inferred for concrete kernels; attribute overrides: `scalar = U`, extra
//! element names (OR'd into the dispatch type-combo), `tiny`/`noalias` (mutually exclusive), and
//! `unroll = N` — the manual knobs over what the `hydroplane-auto` build-time analysis picks.

use hydroplane_auto::analysis;

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{ToTokens, format_ident, quote};
use syn::parse::Parser;
use syn::spanned::Spanned;
use syn::{
    FnArg, GenericParam, Generics, Ident, ItemFn, Meta, Pat, ReturnType, Token, Type,
    TypeParamBound, WherePredicate, parse_macro_input, parse_quote, punctuated::Punctuated,
};

/// Generate a [`Kernel`] (or, with `#[kernel(matrix)]`, a `MatrixKernel`) and a dispatching wrapper
/// from one annotated function. See the [crate docs](crate) for the full shape.
///
/// [`Kernel`]: ../hydroplane/trait.Kernel.html
#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let opts = match KernelOpts::parse(attr.into()) {
        Ok(o) => o,
        Err(e) => return e.to_compile_error().into(),
    };
    match expand(func, opts) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

struct KernelOpts {
    /// `#[kernel(matrix)]`: the context is a [`MatrixBackend`] (still a [`Backend`], so vector ops
    /// work too).
    matrix: bool,
    scalar: Option<Ident>,
    /// Run the body behind a non-inlined `noalias` boundary. On by default; `tiny` turns it off.
    noalias: bool,
    /// `tiny` or `noalias` was written explicitly; build-time analysis must not override it.
    explicit_boundary: bool,
    /// Explicit ILP unroll cap, overriding the build-time estimate. Escape hatch for kernels whose
    /// register pressure the analysis under-counts (e.g. varyings kept in a combinator closure).
    unroll: Option<usize>,
    /// Element names listed in the attribute (`#[kernel(u32, f32)]`): OR'd into the type-combo
    /// bitmask unconditionally, even over MIR-measured sets. The first listed doubles as the
    /// dispatch element when nothing else names one.
    types: Vec<Ident>,
}

fn element_bits(name: &str) -> Option<u8> {
    Some(match name {
        "f32" => 1,
        "f64" => 2,
        "f16" => 4,
        "bf16" => 8,
        "u32" => 16,
        "i32" => 32,
        _ => return None,
    })
}

impl KernelOpts {
    fn parse(tokens: TokenStream2) -> syn::Result<Self> {
        let mut matrix = false;
        let mut scalar = None;
        let mut noalias = true;
        let mut tiny_span = None;
        let mut noalias_span = None;
        let mut unroll = None;
        let mut types = Vec::new();
        let metas = Punctuated::<Meta, Token![,]>::parse_terminated.parse2(tokens)?;
        for meta in metas {
            match meta {
                Meta::Path(p)
                    if p.get_ident().is_some_and(|i| element_bits(&i.to_string()).is_some()) =>
                {
                    types.push(p.get_ident().unwrap().clone());
                }
                Meta::Path(p) if p.is_ident("matrix") => {
                    if matrix {
                        return Err(syn::Error::new(p.span(), "duplicate `matrix`"));
                    }
                    matrix = true;
                }
                Meta::Path(p) if p.is_ident("tiny") => {
                    if tiny_span.is_some() {
                        return Err(syn::Error::new(p.span(), "duplicate `tiny`"));
                    }
                    tiny_span = Some(p.span());
                    noalias = false;
                }
                // Already the default; accepting it explicitly marks the boundary as authored.
                Meta::Path(p) if p.is_ident("noalias") => {
                    if noalias_span.is_some() {
                        return Err(syn::Error::new(p.span(), "duplicate `noalias`"));
                    }
                    noalias_span = Some(p.span());
                }
                Meta::NameValue(nv) if nv.path.is_ident("scalar") => {
                    let id = match &nv.value {
                        syn::Expr::Path(ep) => ep.path.get_ident().cloned(),
                        _ => None,
                    };
                    match id {
                        Some(id) => scalar = Some(id),
                        None => {
                            return Err(syn::Error::new_spanned(
                                &nv.value,
                                "expected an identifier, e.g. `scalar = T`",
                            ));
                        }
                    }
                }
                Meta::NameValue(nv) if nv.path.is_ident("unroll") => match &nv.value {
                    syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(i), .. }) => {
                        unroll = Some(i.base10_parse::<usize>()?);
                    }
                    _ => {
                        return Err(syn::Error::new_spanned(
                            &nv.value,
                            "expected an integer, e.g. `unroll = 4`",
                        ));
                    }
                },
                other => {
                    return Err(syn::Error::new_spanned(
                        other,
                        "unknown `#[kernel]` option; expected `matrix`, `scalar = Ident`, \
                         `unroll = N`, `tiny`, or `noalias`",
                    ));
                }
            }
        }
        if let (Some(ts), Some(ns)) = (tiny_span, noalias_span) {
            let mut e = syn::Error::new(ns, "`noalias` and `tiny` are opposites — pick one");
            e.combine(syn::Error::new(ts, "`tiny` set here"));
            return Err(e);
        }
        let explicit_boundary = tiny_span.is_some() || noalias_span.is_some();
        Ok(KernelOpts { matrix, scalar, noalias, explicit_boundary, unroll, types })
    }
}

fn expand(func: ItemFn, opts: KernelOpts) -> syn::Result<TokenStream2> {
    let func_tokens = func.to_token_stream();
    let ItemFn {
        attrs,
        vis,
        sig,
        block,
    } = func;

    if let Some(a) = sig.asyncness {
        return Err(syn::Error::new_spanned(a, "a kernel cannot be `async`"));
    }
    if let Some(c) = sig.constness {
        return Err(syn::Error::new_spanned(c, "a kernel cannot be `const`"));
    }
    if let Some(v) = sig.variadic {
        return Err(syn::Error::new_spanned(v, "a kernel cannot be variadic"));
    }

    let mut inputs = sig.inputs.into_iter();
    let mut ctx_idents = Vec::with_capacity(1);
    let mut ctx_tys = Vec::with_capacity(1);
    let mut ctx_scalar = None;
    for _ in 0..1 {
        match inputs.next() {
            Some(FnArg::Typed(pt)) => {
                match *pt.pat {
                    Pat::Ident(pi) => ctx_idents.push(pi.ident),
                    other => {
                        return Err(syn::Error::new_spanned(
                            other,
                            "context parameters must be plain identifiers",
                        ));
                    }
                }
                let ty = *pt.ty;
                if ctx_scalar.is_none() {
                    ctx_scalar = ctx_scalar_arg(&ty);
                }
                ctx_tys.push(ctx_type_with_backend(ty)?);
            }
            Some(FnArg::Receiver(r)) => {
                return Err(syn::Error::new_spanned(
                    r,
                    "a kernel takes no `self` receiver",
                ));
            }
            None => {
                return Err(syn::Error::new_spanned(
                    &sig.ident,
                    "a kernel needs a leading context parameter (e.g. `ctx: Gang<T>`) before its \
                     data parameters",
                ));
            }
        }
    }

    let mut field_idents = Vec::new();
    let mut field_tys = Vec::new();
    for arg in inputs {
        match arg {
            FnArg::Typed(pt) => {
                // `#[hint_cnt(N)]` is a slice-length hint, parsed and discarded (reserved for
                // count-driven codegen). Never re-emitted, so it can't reach rustc.
                let _hint_cnt = parse_hint_cnt(&pt.attrs);
                match *pt.pat {
                    Pat::Ident(pi) => field_idents.push(pi.ident),
                    other => {
                        return Err(syn::Error::new_spanned(
                            other,
                            "kernel parameters must be plain identifiers",
                        ));
                    }
                }
                field_tys.push(*pt.ty);
            }
            FnArg::Receiver(r) => {
                return Err(syn::Error::new_spanned(
                    r,
                    "a kernel takes no `self` receiver",
                ));
            }
        }
    }

    let ret: Type = match sig.output {
        ReturnType::Default => parse_quote!(()),
        ReturnType::Type(_, t) => *t,
    };

    // Scalar precedence: explicit `scalar = …` > `Scalar`-bound or `T`-named generic > the
    // context's element type > first attribute element > lowest scanned element > `f32`. The
    // element only keys the `Kernel<T>` trait, so any element the kernel touches is correct.
    let scanned_bits_early = {
        let mut ts = TokenStream2::new();
        func_tokens.to_tokens(&mut ts);
        scan_type_bits(&ts)
    };
    let detected = detect_scalar(&sig.generics)?;
    let scalar = if let Some(s) = &opts.scalar {
        quote!(#s)
    } else if let Some(s) = &detected {
        quote!(#s)
    } else if let Some(t) = &ctx_scalar {
        quote!(#t)
    } else if let Some(first) = opts.types.first() {
        quote!(#first)
    } else {
        match 1u8 << scanned_bits_early.trailing_zeros().min(6) {
            1 => quote!(f32),
            2 => quote!(f64),
            4 => quote!(::hydroplane::f16),
            8 => quote!(::hydroplane::bf16),
            16 => quote!(u32),
            32 => quote!(i32),
            _ => quote!(f32),
        }
    };

    let generics = sig.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let mut lifetimes = Vec::new();
    let mut type_params = Vec::new();
    let mut turbofish_args = Vec::new();
    for p in &generics.params {
        match p {
            GenericParam::Lifetime(l) => lifetimes.push(l.lifetime.clone()),
            GenericParam::Type(t) => {
                type_params.push(t.ident.clone());
                turbofish_args.push(t.ident.clone());
            }
            GenericParam::Const(c) => turbofish_args.push(c.ident.clone()),
        }
    }

    // PhantomData over every lifetime and type parameter keeps the struct well-formed when a
    // parameter only appears in the return type or a const dimension; `*const T` adds no
    // Send/Sync/Drop obligation. Unused const generics are allowed, so no marker for them.
    let phantom_items: Vec<TokenStream2> = lifetimes
        .iter()
        .map(|lt| quote!(& #lt ()))
        .chain(type_params.iter().map(|t| quote!(*const #t)))
        .collect();
    let (phantom_field, phantom_init) = if phantom_items.is_empty() {
        (quote!(), quote!())
    } else {
        (
            quote!(pub(super) __phantom: ::core::marker::PhantomData<(#(#phantom_items),*)>,),
            quote!(__phantom: ::core::marker::PhantomData,),
        )
    };

    let matrix_present = opts.matrix;
    let (ktrait, bbound, dispatch_fn, dbound) = if matrix_present {
        (
            quote!(::hydroplane::MatrixKernel),
            quote!(::hydroplane::MatrixBackend),
            quote!(::hydroplane::dispatch_matrix),
            quote!(::hydroplane::MatrixDispatch),
        )
    } else {
        (
            quote!(::hydroplane::Kernel),
            quote!(::hydroplane::Backend),
            quote!(::hydroplane::dispatch),
            quote!(::hydroplane::SimdDispatch),
        )
    };

    let run_bound = if matrix_present {
        quote!(#bbound<#scalar>)
    } else {
        quote!(::hydroplane::BackendAll + ::hydroplane::Backend<#scalar>)
    };

    let mut wrapper_generics = generics.clone();
    let pred: WherePredicate = parse_quote!(#scalar: #dbound);
    wrapper_generics.make_where_clause().predicates.push(pred);
    let (w_impl_generics, _, w_where) = wrapper_generics.split_for_impl();

    let turbofish = if turbofish_args.is_empty() {
        quote!()
    } else {
        quote!(::<#(#turbofish_args),*>)
    };

    let name = sig.ident;

    // Fold build-time MIR analysis into the codegen knobs. Matrix kernels dispatch through a
    // different path (no `Unroll` wrapper, no `run_scalar`), so they take none of it; an explicit
    // `tiny`/`noalias` always wins over the analysis boundary choice.
    let name_str = name.to_string();
    let metrics = if matrix_present {
        None
    } else {
        analysis::lookup(&name_str)
    };
    let use_noalias = match metrics {
        Some(m) if !opts.explicit_boundary => m.noalias(),
        _ => opts.noalias,
    };
    let k_cap = opts.unroll.or_else(|| metrics.map(|m| m.k_cap()));

    // Under the default `noalias` boundary the real body lives in `_on`, which is what the analysis
    // driver measures; explicit-`tiny` and matrix kernels aren't measured. The metrics attribute is
    // emitted only inside the driver's nested build (`HYDRO_ANALYZE_INNER` + `--cfg hp_analyze`):
    // a normal consumer build must never see the `hp_analyze` token, because `unexpected_cfgs`
    // fires before any generated `#[allow]` could apply, so keeping the `cfg_attr` out entirely is
    // the only way consumers stay warning-free without registering the cfg themselves.
    let metrics_attr = if use_noalias
        && !matrix_present
        && std::env::var_os("HYDRO_ANALYZE_INNER").is_some()
    {
        quote!(#[cfg_attr(hp_analyze, hp_analyze::metrics(kernel = #name_str))])
    } else {
        quote!()
    };
    let k_cap_item = match k_cap {
        Some(k) => quote!(const K_CAP: usize = #k;),
        None => quote!(),
    };

    // The trait's `run` takes one context; any extra contexts become `Copy` aliases of it so each
    // keeps its declared name inside the body.
    let primary_ctx = &ctx_idents[0];
    let primary_ty = &ctx_tys[0];
    let ctx_aliases = ctx_idents[1..]
        .iter()
        .map(|id| quote!(let #id = #primary_ctx;));

    // `<name>_on(ctx, …)` runs the body on a caller-supplied context, skipping dispatch, so nested
    // kernel calls dispatch once at the outer boundary. Generic over the backend `__S`, no dispatch
    // bound.
    let on_name = format_ident!("{}_on", name);
    let mut on_generics = generics.clone();
    on_generics.params.push(syn::GenericParam::Type(parse_quote!(__S: #run_bound)));
    let (on_impl_generics, _, on_where) = on_generics.split_for_impl();

    // Default (`!tiny`): the body runs inside `_on`, whose slice parameters keep the `noalias`
    // attribute that a load through the `__Kernel` struct field drops. The attribute only survives
    // a genuine call, so `_on` is `#[inline(never)]`; `tiny` inlines everything instead.
    let on_call_turbofish = if turbofish_args.is_empty() {
        quote!(::<__S>)
    } else {
        quote!(::<#(#turbofish_args,)* __S>)
    };
    let (run_body, on_body, on_inline) = if use_noalias {
        (
            quote! {
                let Self { #( #field_idents, )* .. } = self;
                #on_name #on_call_turbofish ( #primary_ctx, #( #field_idents, )* )
            },
            quote! {
                #( #ctx_aliases )*
                #block
            },
            quote!(#[inline(never)]),
        )
    } else {
        (
            quote! {
                #( #ctx_aliases )*
                let Self { #( #field_idents, )* .. } = self;
                #block
            },
            quote! {
                #ktrait::run(#name::__Kernel #turbofish { #( #field_idents, )* #phantom_init }, #primary_ctx)
            },
            quote!(#[inline]),
        )
    };

    // Type-combo: the scanned (or MIR-measured) element set plus the dispatch element's own bits,
    // baked as a constant in the wrapper.
    let attr_bits: u8 = opts
        .types
        .iter()
        .filter_map(|i| element_bits(&i.to_string()))
        .fold(0, |a, b| a | b);
    let combo_bits = metrics.and_then(|m| m.type_bits()).unwrap_or(scanned_bits_early) | attr_bits;
    // A generic scalar's tier arms cannot be pruned; the element is only known at monomorphization.
    let scalar_str = quote!(#scalar).to_string();
    let scalar_is_generic = generics.params.iter().any(|p| match p {
        GenericParam::Type(t) => t.ident == scalar_str,
        _ => false,
    });
    let may_half_f16 = combo_bits & 4 != 0;
    let may_half_bf16 = combo_bits & 8 != 0;
    let pure_float = combo_bits & 60 == 0;
    let no_ints = combo_bits & 48 == 0;

    let fp16_arm = may_half_f16.then(|| quote! {
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        // SAFETY (all arms): `combo_tier` returns a tier code only after feature detection (or a
        // compile-time guarantee) confirmed the host supports it.
        ::hydroplane::towers::AVX512FP16 => ::hydroplane::dispatch::run_kernel_on(
            __kernel,
            unsafe { ::hydroplane::towers::Avx512Fp16::new_unchecked() },
        ),
    });
    let bf16_arm = may_half_bf16.then(|| quote! {
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        ::hydroplane::towers::AVX512BF16 => ::hydroplane::dispatch::run_kernel_on(
            __kernel,
            unsafe { ::hydroplane::towers::Avx512Bf16::new_unchecked() },
        ),
    });
    let avx1_arm = pure_float.then(|| quote! {
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        ::hydroplane::towers::AVX1 => ::hydroplane::dispatch::run_kernel_on(
            __kernel,
            unsafe { ::hydroplane::towers::Avx1::new_unchecked() },
        ),
    });
    let sve_arms = no_ints.then(|| quote! {
        #[cfg(all(target_arch = "aarch64", not(target_vendor = "apple")))]
        ::hydroplane::towers::SVE16 => ::hydroplane::dispatch::run_kernel_on(
            __kernel,
            unsafe { ::hydroplane::towers::Sve::<16>::new_unchecked() },
        ),
        #[cfg(all(target_arch = "aarch64", not(target_vendor = "apple")))]
        ::hydroplane::towers::SVE32 => ::hydroplane::dispatch::run_kernel_on(
            __kernel,
            unsafe { ::hydroplane::towers::Sve::<32>::new_unchecked() },
        ),
        #[cfg(all(target_arch = "aarch64", not(target_vendor = "apple")))]
        ::hydroplane::towers::SVE64 => ::hydroplane::dispatch::run_kernel_on(
            __kernel,
            unsafe { ::hydroplane::towers::Sve::<64>::new_unchecked() },
        ),
    });

    // Combo dispatch needs `Token: Backend<T>` provable in the wrapper body, which only holds for
    // concrete elements; generic-scalar kernels keep the element-keyed `dispatch` path.
    let dispatch_call = if matrix_present || scalar_is_generic {
        quote!(#dispatch_fn(#name::__Kernel #turbofish { #( #field_idents, )* #phantom_init }))
    } else {
        quote! {{
            let __combo: u8 = #combo_bits | <#scalar as ::hydroplane::Scalar>::TYPE_BITS;
            let __kernel = #name::__Kernel #turbofish { #( #field_idents, )* #phantom_init };
            match ::hydroplane::dispatch::combo_tier(__combo) {
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                ::hydroplane::towers::SSE4 => ::hydroplane::dispatch::run_kernel_on(
                    __kernel,
                    unsafe { ::hydroplane::towers::Sse4::new_unchecked() },
                ),
                #avx1_arm
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                ::hydroplane::towers::AVX2 => ::hydroplane::dispatch::run_kernel_on(
                    __kernel,
                    unsafe { ::hydroplane::towers::Avx2::new_unchecked() },
                ),
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                ::hydroplane::towers::AVX512 => ::hydroplane::dispatch::run_kernel_on(
                    __kernel,
                    unsafe { ::hydroplane::towers::Avx512::new_unchecked() },
                ),
                #fp16_arm
                #bf16_arm
                #[cfg(target_arch = "aarch64")]
                ::hydroplane::towers::NEON => ::hydroplane::dispatch::run_kernel_on(
                    __kernel,
                    ::hydroplane::towers::Neon::new(),
                ),
                #sve_arms
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128", target_feature = "relaxed-simd"))]
                ::hydroplane::towers::RELAXED => ::hydroplane::dispatch::run_kernel_on(
                    __kernel,
                    ::hydroplane::towers::RelaxedSimd::new(),
                ),
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                ::hydroplane::towers::SIMD128 => ::hydroplane::dispatch::run_kernel_on(
                    __kernel,
                    ::hydroplane::towers::Simd128::new(),
                ),
                _ => ::hydroplane::dispatch::run_kernel_on(__kernel, ::hydroplane::ScalarBackend),
            }
        }}
    };

    Ok(quote! {
        #[doc(hidden)]
        #[allow(non_snake_case)]
        mod #name {
            #[allow(unused_imports)]
            use super::*;
            #[allow(dead_code)]
            pub(super) struct __Kernel #impl_generics #where_clause {
                #( pub(super) #field_idents: #field_tys, )*
                #phantom_field
            }
            impl #impl_generics #ktrait<#scalar> for __Kernel #ty_generics #where_clause {
                type Output = #ret;
                #k_cap_item
                #[inline]
                fn run<__S: #run_bound>(self, #primary_ctx: #primary_ty) -> #ret {
                    #run_body
                }
            }
        }
        #(#attrs)*
        #[inline]
        #[allow(clippy::multiple_bound_locations)]
        // Splat-constant scalars arrive as plain params, so kernels routinely exceed clippy's arg
        // limit; the author can't annotate a generated fn.
        #[allow(clippy::too_many_arguments)]
        #vis fn #name #w_impl_generics ( #( #field_idents: #field_tys ),* ) -> #ret
        #w_where
        {
            #dispatch_call
        }
        #[doc = concat!("Run [`", stringify!(#name), "`]'s body on an already-dispatched context, ")]
        #[doc = "skipping a second dispatch — call this from inside another kernel to reuse its backend."]
        #metrics_attr
        #on_inline
        #[allow(clippy::too_many_arguments)]
        #vis fn #on_name #on_impl_generics ( #primary_ctx: #primary_ty, #( #field_idents: #field_tys ),* ) -> #ret
        #on_where
        {
            #on_body
        }
    })
}

/// Element bits (`Scalar::TYPE_BITS` values) whose names appear anywhere in the kernel's tokens: a
/// safe over-approximation of the elements the body touches, used to prune the combo-dispatch
/// match. MIR analysis replaces this with the measured set when available; the generated combo
/// also ORs the dispatch element's own `TYPE_BITS`, so the declared element is never missed.
fn scan_type_bits(tokens: &TokenStream2) -> u8 {
    fn walk(ts: TokenStream2, bits: &mut u8) {
        for tt in ts {
            match tt {
                proc_macro2::TokenTree::Ident(id) => {
                    *bits |= match id.to_string().as_str() {
                        "f32" => 1,
                        "f64" => 2,
                        "f16" => 4,
                        "bf16" => 8,
                        "u32" => 16,
                        "i32" => 32,
                        _ => 0,
                    };
                }
                proc_macro2::TokenTree::Group(g) => walk(g.stream(), bits),
                _ => {}
            }
        }
    }
    let mut bits = 0;
    walk(tokens.clone(), &mut bits);
    bits
}

/// The `#[hint_cnt(N)]` value on a kernel parameter, if present. Currently discarded by the caller;
/// malformed forms are leniently ignored.
fn parse_hint_cnt(attrs: &[syn::Attribute]) -> Option<u64> {
    attrs
        .iter()
        .filter(|a| a.path().is_ident("hint_cnt"))
        .find_map(|a| a.parse_args::<syn::LitInt>().ok())
        .and_then(|lit| lit.base10_parse::<u64>().ok())
}

/// Rewrite the context parameter's written type into the backend-carrying form, filling in `__S`.
/// The author's path is preserved, so a plain `Gang` import and a fully-qualified
/// `hydroplane::Gang<T>` both work.
fn ctx_type_with_backend(ty: Type) -> syn::Result<Type> {
    let bad = |t: &dyn quote::ToTokens| {
        syn::Error::new_spanned(
            t,
            "the context parameter must be typed `Gang<T>` (the scalar); the backend type argument \
             is filled in for you",
        )
    };
    let Type::Path(mut tp) = ty else {
        return Err(bad(&ty));
    };
    let Some(seg) = tp.path.segments.last_mut() else {
        return Err(bad(&tp));
    };
    // Bare `ctx: Gang` is canonical (the gang is element-free); a legacy bracketed element
    // (`Gang<f32>`) is accepted and names the dispatch element.
    match &mut seg.arguments {
        syn::PathArguments::None => {
            seg.arguments = syn::PathArguments::AngleBracketed(parse_quote!(<__S>));
        }
        syn::PathArguments::AngleBracketed(args) => {
            args.args.clear();
            args.args.push(parse_quote!(__S));
        }
        _ => return Err(bad(&tp)),
    }
    Ok(Type::Path(tp))
}

/// First type argument of the context type (`Gang<f32>` → `f32`), used to infer the scalar of a
/// kernel with no generic scalar parameter.
fn ctx_scalar_arg(ty: &Type) -> Option<Type> {
    let Type::Path(tp) = ty else { return None };
    let seg = tp.path.segments.last()?;
    let syn::PathArguments::AngleBracketed(ab) = &seg.arguments else {
        return None;
    };
    ab.args.iter().find_map(|a| match a {
        syn::GenericArgument::Type(t) => Some(t.clone()),
        _ => None,
    })
}

/// Find the scalar type parameter: the one bound by a trait named `Scalar` (inline or in the
/// where-clause), falling back to a parameter literally named `T`. `Ok(None)` if there is no scalar
/// type parameter at all (a concrete-scalar kernel); `Err` only when the choice is ambiguous.
fn detect_scalar(generics: &Generics) -> syn::Result<Option<Ident>> {
    let has_scalar_bound = |bounds: &Punctuated<TypeParamBound, Token![+]>| -> bool {
        bounds.iter().any(|b| match b {
            TypeParamBound::Trait(tb) => tb.path.segments.last().is_some_and(|s| {
                s.ident == "Scalar" || s.ident == "FloatScalar" || s.ident == "IntScalar"
            }),
            _ => false,
        })
    };

    let mut scalar_bound: Option<Ident> = None;
    let mut named_t: Option<Ident> = None;
    let mut record = |id: &Ident, span: Span| -> syn::Result<()> {
        if let Some(prev) = &scalar_bound
            && prev != id
        {
            return Err(syn::Error::new(
                span,
                "multiple type parameters are bound by `Scalar`; disambiguate with `#[kernel(scalar = ...)]`",
            ));
        }
        scalar_bound = Some(id.clone());
        Ok(())
    };

    for p in &generics.params {
        if let GenericParam::Type(tp) = p {
            if tp.ident == "T" {
                named_t = Some(tp.ident.clone());
            }
            if has_scalar_bound(&tp.bounds) {
                record(&tp.ident, tp.ident.span())?;
            }
        }
    }
    if let Some(wc) = &generics.where_clause {
        for pred in &wc.predicates {
            if let WherePredicate::Type(pt) = pred
                && let Type::Path(typ) = &pt.bounded_ty
                && let Some(id) = typ.path.get_ident()
                && has_scalar_bound(&pt.bounds)
            {
                record(id, id.span())?;
            }
        }
    }

    Ok(scalar_bound.or(named_t))
}
