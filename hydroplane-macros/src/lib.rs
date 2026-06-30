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
//! ## Modes
//!
//! The attribute lists the execution surfaces the kernel needs, in the order their context
//! parameters appear:
//!
//! * `#[kernel(vector)]` (the default, also written bare `#[kernel]`) — one [`Backend<T>`] context.
//! * `#[kernel(matrix)]` — one [`MatrixBackend<T>`] context (adds the `.tiles()` matmul surface).
//! * `#[kernel(vector, matrix)]` — two leading contexts in that order, both over the *same*
//!   dispatched backend (so the first is a plain vector handle, the second a matrix handle).
//!
//! Whenever `matrix` is requested the kernel is a [`MatrixKernel`] dispatched via `dispatch_matrix`
//! and every context is bound by `MatrixBackend<T>` (which is itself a `Backend<T>`, so the vector
//! handle works too); otherwise it is a [`Kernel`] dispatched via `dispatch`.
//!
//! ```ignore
//! #[hydroplane::kernel(vector, matrix)]
//! fn gemv<'a, T: Scalar, const M: usize, const K: usize>(
//!     v: Gang<T>, m: Gang<T>, a: &'a [T], x: &'a [T], out: &'a mut [T::Compute],
//! ) {
//!     let _ = v.lanes();        // vector handle
//!     let _tiles = m.tiles();   // matrix handle, same backend
//! }
//! ```
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

/// A requested execution surface. Each mode contributes one leading context parameter, in the order
/// listed in the attribute, and determines the backend bound: [`Vector`](Mode::Vector) needs
/// `Backend<T>`, [`Matrix`](Mode::Matrix) needs `MatrixBackend<T>`. The kernel is dispatched through
/// the most capable surface requested (matrix when present, since `MatrixBackend<T>: Backend<T>`),
/// and every context parameter shares that one backend.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Vector,
    Matrix,
}

struct KernelOpts {
    modes: Vec<Mode>,
    scalar: Option<Ident>,
}

impl KernelOpts {
    fn parse(tokens: TokenStream2) -> syn::Result<Self> {
        let mut modes = Vec::new();
        let mut scalar = None;
        let metas = Punctuated::<Meta, Token![,]>::parse_terminated.parse2(tokens)?;
        for meta in metas {
            let push_mode = |modes: &mut Vec<Mode>, m: Mode, span| -> syn::Result<()> {
                if modes.contains(&m) {
                    return Err(syn::Error::new(span, "duplicate mode"));
                }
                modes.push(m);
                Ok(())
            };
            match meta {
                Meta::Path(p) if p.is_ident("vector") => {
                    push_mode(&mut modes, Mode::Vector, p.span())?
                }
                Meta::Path(p) if p.is_ident("matrix") => {
                    push_mode(&mut modes, Mode::Matrix, p.span())?
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
                other => {
                    return Err(syn::Error::new_spanned(
                        other,
                        "unknown `#[kernel]` option; expected `vector`, `matrix`, or `scalar = Ident`",
                    ));
                }
            }
        }
        // Bare `#[kernel]` (or `#[kernel(scalar = …)]` alone) means a single vector context.
        if modes.is_empty() {
            modes.push(Mode::Vector);
        }
        Ok(KernelOpts { modes, scalar })
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

    let n_ctx = opts.modes.len();
    let mut inputs = sig.inputs.into_iter();
    let mut ctx_idents = Vec::with_capacity(n_ctx);
    let mut ctx_tys = Vec::with_capacity(n_ctx);
    let mut ctx_scalar = None;
    for _ in 0..n_ctx {
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
                    format!(
                        "this kernel declares {n_ctx} mode(s), so it needs {n_ctx} leading context \
                         parameter(s) (e.g. `ctx: Gang<T>`) before its data parameters",
                    ),
                ));
            }
        }
    }

    let mut field_idents = Vec::new();
    let mut field_tys = Vec::new();
    for arg in inputs {
        match arg {
            FnArg::Typed(pt) => {
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

    let matrix_present = opts.modes.contains(&Mode::Matrix);
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
                #[inline]
                fn run<__S: #bbound<#scalar>>(self, #primary_ctx: #primary_ty) -> #ret {
                    #( #ctx_aliases )*
                    let Self { #( #field_idents, )* .. } = self;
                    #block
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
            #dispatch_fn(#name::__Kernel #turbofish { #( #field_idents, )* #phantom_init })
        }
        #[doc = concat!("Run [`", stringify!(#name), "`]'s body on an already-dispatched context, ")]
        #[doc = "skipping a second dispatch — call this from inside another kernel to reuse its backend."]
        #[inline]
        #[allow(clippy::too_many_arguments)]
        #vis fn #on_name #on_impl_generics ( #primary_ctx: #primary_ty, #( #field_idents: #field_tys ),* ) -> #ret
        #on_where
        {
            #ktrait::run(#name::__Kernel #turbofish { #( #field_idents, )* #phantom_init }, #primary_ctx)
        }
    })
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
