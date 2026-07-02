//! The `#[kernel]` attribute — write a `hydroplane` kernel as a plain generic function.
//!
//! A kernel must be generic over the backend (unknown until runtime dispatch picks it), and a Rust
//! closure can't be generic over a type, so the hand-written form is a struct (to carry the borrows
//! past the dispatch boundary) plus a [`Kernel`] impl with a generic `run` method. `#[kernel]`
//! generates both from a function whose first parameter is the SIMD context, and emits a callable
//! that runs `dispatch` for you:
//!
//! ```ignore
//! #[hydroplane::kernel]
//! pub fn any_overlap<'a, T: Scalar>(ctx: Gang<T>, soa: &'a Soa<T>, q: [T; 4]) -> bool {
//!     // `ctx`, `soa`, `q` are all in scope; write the kernel body directly.
//! }
//! // call site — no struct, no impl, no `dispatch`:
//! let hit = any_overlap(&soa, q);
//! ```
//!
//! The leading parameters are contexts: annotate each `Gang<T>` (the backend is filled in by
//! dispatch, so the second type argument is left off). Every parameter after the contexts becomes a
//! field carried across the dispatch boundary. The scalar type parameter is the one bound by
//! `Scalar` (or, failing that, the one named `T`); override the choice with `#[kernel(scalar = U)]`.
//!
//! ## Surface
//!
//! A kernel takes exactly one leading context parameter (`ctx: Gang<T>`). Bare `#[kernel]` binds it to
//! a [`Backend<T>`] — the element-wise SIMD surface (`splat`/`load`/`map`/`sum`…). `#[kernel(matrix)]`
//! binds it to a [`MatrixBackend<T>`] instead, adding the `.tiles()` matmul surface — and since
//! `MatrixBackend<T>: Backend<T>`, a matrix context still does every vector op, so one `ctx` serves
//! both. A matrix kernel is a [`MatrixKernel`] dispatched via `dispatch_matrix`; a vector kernel is a
//! [`Kernel`] dispatched via `dispatch`.
//!
//! ```ignore
//! #[hydroplane::kernel(matrix)]
//! fn gemm<'a, T: Scalar, const M: usize, const N: usize, const K: usize>(
//!     ctx: Gang<T>, a: &'a [T], b: &'a [T], out: &'a mut [T::Compute],
//! ) {
//!     let tl = ctx.tiles();                                   // matmul surface
//!     tl.mma::<M, N, K>(tl.load_a_rm::<M, K>(a), tl.load_b_rm::<K, N>(b), tl.zero_acc::<M, N>())
//!         .store_rm(out);
//!     let _ = ctx.lanes();                                    // …and vector ops on the same ctx
//! }
//! ```
//!
//! ## Tuning
//!
//! By default the body runs behind a non-inlined `noalias` boundary: its data parameters are passed
//! as real function arguments, so `&`/`&mut` slices carry the `noalias` attribute a load through the
//! generated kernel struct would drop — worth up to ~1.6x on memory-bound kernels by letting LLVM
//! cluster and reorder loads/stores. The trade is a single per-invocation call. `#[kernel(tiny)]`
//! opts out (fully inlined, no call) for micro-kernels whose fixed work is smaller than that call;
//! `#[kernel(noalias)]` states the default explicitly. `tiny` and `noalias` are mutually exclusive.
//! `#[kernel(unroll = N)]` pins the ILP unroll factor. `noalias`/`tiny` and `unroll` are also what the
//! build-time analysis (`hydroplane-auto`) chooses automatically; the attribute is the manual override.
//!
//! The scalar may be concrete: `#[kernel] fn any_gt(ctx: Gang<f32>, …)` infers `f32` from the context
//! type, so no generic parameter is required. Backend selection inside `dispatch` is itself cached
//! (resolved once per process), so even tiny kernels called in a hot loop don't re-probe the CPU.
//!
//! ## Calling a kernel from another kernel
//!
//! Every kernel also gets a `<name>_on(ctx, …)` companion that runs the body on a context **you**
//! supply, skipping dispatch. Call it from inside another kernel to reuse the outer kernel's
//! already-dispatched backend, so dispatch happens once (at the outer boundary) instead of again per
//! inner call:
//!
//! ```ignore
//! #[hydroplane::kernel]
//! fn scaled<'a, T: Scalar>(ctx: Gang<T>, xs: &'a [T], k: T) -> f64 { /* … */ }
//!
//! #[hydroplane::kernel]
//! fn scaled_then_sum<'a, T: Scalar>(ctx: Gang<T>, xs: &'a [T], ys: &'a [T], k: T) -> f64 {
//!     scaled_on(ctx, xs, k) + scaled_on(ctx, ys, k)   // one dispatch, two inner runs
//! }
//! ```
//!
//! Unlike the `macro_rules!` fallback, generics use ordinary `<…>` syntax and may carry multiple
//! bounds, where-clauses, several lifetimes, and several type parameters.
//!
//! [`Backend<T>`]: ../hydroplane/trait.Backend.html
//! [`MatrixBackend<T>`]: ../hydroplane/trait.MatrixBackend.html
//!
//! [`Kernel`]: ../hydroplane/trait.Kernel.html
//! [`MatrixKernel`]: ../hydroplane/trait.MatrixKernel.html

use hydroplane_auto::analysis;

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
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
    /// `#[kernel(matrix)]` — the single context is a [`MatrixBackend`], adding the `.tiles()` matmul
    /// surface (and it is still a [`Backend`], so vector ops work too). Bare `#[kernel]` is a plain
    /// vector context.
    matrix: bool,
    scalar: Option<Ident>,
    /// Run the body behind a non-inlined `noalias` boundary. On by default (worth up to ~1.6x on
    /// memory-bound kernels); `tiny` turns it off for micro-kernels that can't afford the call.
    noalias: bool,
    /// The author wrote `tiny` or `noalias` explicitly. When set, build-time analysis must not
    /// override the boundary choice — an explicit annotation always wins.
    explicit_boundary: bool,
    /// Explicit ILP unroll cap, overriding the build-time estimate. The escape hatch for kernels
    /// whose register pressure the analysis can't measure — a `map_cols` transform that keeps its
    /// varyings in a closure the register proxy under-counts, so the author pins `unroll = 1`.
    unroll: Option<usize>,
}

impl KernelOpts {
    fn parse(tokens: TokenStream2) -> syn::Result<Self> {
        let mut matrix = false;
        let mut scalar = None;
        let mut noalias = true;
        let mut tiny_span = None;
        let mut noalias_span = None;
        let mut unroll = None;
        let metas = Punctuated::<Meta, Token![,]>::parse_terminated.parse2(tokens)?;
        for meta in metas {
            match meta {
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
                // The `noalias` boundary is the default; `noalias` is accepted as an explicit
                // affirmation, so a kernel can state its intent (and to keep older annotations valid).
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
        Ok(KernelOpts { matrix, scalar, noalias, explicit_boundary, unroll })
    }
}

fn expand(func: ItemFn, opts: KernelOpts) -> syn::Result<TokenStream2> {
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

    // A kernel takes exactly one leading context parameter (`ctx: Gang<T>`).
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
                // `#[hint_cnt(N)]` on a data parameter is a length hint for the passed slice/iterator.
                // Parsed and thrown out for now — reserved for future count-driven codegen (static tail
                // specialization, prefetch distance). It is never re-emitted, so it can't reach rustc.
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

    // Scalar precedence: explicit `scalar = …` > the `Scalar`-bound / `T`-named generic > the
    // context's element type (lets a concrete `ctx: Gang<f32>` kernel skip a generic parameter).
    let detected = detect_scalar(&sig.generics)?;
    let scalar = if let Some(s) = &opts.scalar {
        quote!(#s)
    } else if let Some(s) = &detected {
        quote!(#s)
    } else if let Some(t) = &ctx_scalar {
        quote!(#t)
    } else {
        return Err(syn::Error::new_spanned(
            &sig.ident,
            "could not determine the scalar type: bind a generic with `Scalar` (e.g. `T: Scalar`), \
             give the context an explicit element (`ctx: Gang<f32>`), or pass `#[kernel(scalar = X)]`",
        ));
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

    // A PhantomData over every lifetime and type parameter keeps the struct well-formed even when a
    // parameter only appears in the return type or a const dimension — `*const T` so the marker adds
    // no Send/Sync/Drop obligation. Const params need no marker (unused const generics are allowed).
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

    // Fold build-time MIR analysis (when available) into the codegen knobs. Matrix kernels dispatch
    // through a different path (`dispatch_matrix`, no `Unroll` wrapper, no `run_scalar`), so they take
    // none of it. An explicit `tiny`/`noalias` always wins over the analysis boundary choice.
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
    // An explicit `unroll = N` pins the cap; otherwise the build-time estimate (if any) sets it.
    let k_cap = opts.unroll.or_else(|| metrics.map(|m| m.k_cap()));

    // Under the (default) `noalias` boundary the real body lives in `_on`, so that is what the analysis
    // driver measures. The attribute is inert until the driver compiles with `--cfg hydro_analyze`.
    // Explicit-`tiny` and matrix kernels aren't measured (the body isn't in `_on`, or isn't capped).
    // Only the analysis driver's nested build (which sets `HYDRO_ANALYZE_INNER` alongside
    // `--cfg hydro_analyze`) ever evaluates the metrics attribute, so it is emitted only there.
    // A normal consumer build never sees the `hydro_analyze` token: the `unexpected_cfgs` lint
    // fires during expansion config, before any generated `#[allow]` could apply, so keeping the
    // `cfg_attr` out entirely is the only way consumers stay warning-free without registering
    // the cfg in their own check-cfg table.
    let metrics_attr = if use_noalias
        && !matrix_present
        && std::env::var_os("HYDRO_ANALYZE_INNER").is_some()
    {
        quote!(#[cfg_attr(hydro_analyze, hydro_analyze::metrics(kernel = #name_str))])
    } else {
        quote!()
    };
    let k_cap_item = match k_cap {
        Some(k) => quote!(const K_CAP: usize = #k;),
        None => quote!(),
    };

    // The trait's `run` takes one context. Every requested mode shares the single dispatched backend,
    // so the first context is the real parameter and the rest are `Copy` aliases of it (`Gang` is
    // `Copy`), giving each its declared name and surface inside the body.
    let primary_ctx = &ctx_idents[0];
    let primary_ty = &ctx_tys[0];
    let ctx_aliases = ctx_idents[1..]
        .iter()
        .map(|id| quote!(let #id = #primary_ctx;));

    // `<name>_on(ctx, …)` runs the body on a caller-supplied, already-dispatched context — the entry
    // point one kernel calls from inside another so the backend is selected only once, at the outer
    // kernel's boundary, instead of re-dispatching per inner call. Generic over the backend `__S`
    // (no dispatch bound); takes the primary context plus the data parameters.
    let on_name = format_ident!("{}_on", name);
    let mut on_generics = generics.clone();
    on_generics.params.push(parse_quote!(__S: #bbound<#scalar>));
    let (on_impl_generics, _, on_where) = on_generics.split_for_impl();

    // Default (`!tiny`): run the body inside `_on` — whose data parameters are real function
    // arguments, so the `&`/`&mut` slices carry the `noalias` attribute that a load through the
    // `__Kernel` struct field drops. That attribute lets LLVM cluster and reorder loads/stores across
    // the loop (worth up to ~1.6x on memory-bound kernels), but only survives a genuine call, so `_on`
    // is `#[inline(never)]`. `#[kernel(tiny)]` opts out — fully inlined, no per-invocation call — for
    // micro-kernels whose fixed work is smaller than that call (~0.2ns).
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

    let dispatch_call =
        quote!(#dispatch_fn(#name::__Kernel #turbofish { #( #field_idents, )* #phantom_init }));

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
                fn run<__S: #bbound<#scalar>>(self, #primary_ctx: #primary_ty) -> #ret {
                    #run_body
                }
            }
        }
        #(#attrs)*
        #[inline]
        #[allow(clippy::multiple_bound_locations)]
        // Splat-constant scalars arrive as plain params, so kernels routinely exceed clippy's arg
        // limit; the author can't annotate a generated fn, so the wrapper opts out for them.
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

/// The value of a `#[hint_cnt(N)]` attribute on a kernel parameter, if present. Currently discarded
/// by the caller (the hint is not yet wired into codegen); malformed forms are leniently ignored.
fn parse_hint_cnt(attrs: &[syn::Attribute]) -> Option<u64> {
    attrs
        .iter()
        .filter(|a| a.path().is_ident("hint_cnt"))
        .find_map(|a| a.parse_args::<syn::LitInt>().ok())
        .and_then(|lit| lit.base10_parse::<u64>().ok())
}

/// Turn the context parameter's written type into the backend-carrying form: `Gang<T>` (written by
/// the author, with the backend elided) becomes `Gang<T, __S>`. The author's path is preserved, so a
/// plain `Gang` import stays meaningful and a fully-qualified `hydroplane::Gang<T>` works too.
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
    let syn::PathArguments::AngleBracketed(args) = &mut seg.arguments else {
        return Err(bad(&tp));
    };
    args.args.push(parse_quote!(__S));
    Ok(Type::Path(tp))
}

/// The first type argument of the context type — `Gang<f32>` → `f32`, `Gang<T>` → `T` — used to
/// infer the scalar of a kernel that has no generic scalar parameter (a concrete `Gang<f32>` kernel).
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
            TypeParamBound::Trait(tb) => {
                tb.path.segments.last().is_some_and(|s| s.ident == "Scalar")
            }
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
