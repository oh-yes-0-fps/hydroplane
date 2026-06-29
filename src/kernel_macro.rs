//! The [`kernel!`] macro — collapse the [`Kernel`](crate::Kernel)/[`MatrixKernel`](crate::MatrixKernel)
//! struct-plus-impl boilerplate into one annotated function.
//!
//! A kernel must be generic over the backend `S` (unknown until runtime dispatch picks it), and a
//! Rust closure can't be generic over a type, so the hand-written form is a struct (to carry the
//! borrows past the dispatch boundary) plus a `Kernel` impl with a generic `run` method. `kernel!`
//! generates both from a function whose first parameter is the SIMD context, and emits a callable
//! that runs `dispatch` for you:
//!
//! ```ignore
//! hydroplane::kernel! {
//!     /// Any sphere overlapping the query?
//!     pub fn any_overlap['a, T: Scalar](ctx, soa: &'a Soa<T>, q: [T; 4]) -> bool {
//!         // `ctx`, `soa`, `q` are all in scope; write the kernel body directly.
//!     }
//! }
//! // call site — no struct, no impl, no `dispatch`:
//! let hit = any_overlap(&soa, q);
//! ```
//!
//! Generics go in **square brackets** `[…]` (a token-tree group — `<…>` is ambiguous to
//! `macro_rules`); the generated `fn`/`impl` use normal `<…>`. Use `matrix fn` for a
//! [`MatrixKernel`](crate::MatrixKernel) (the context backend is then
//! [`MatrixBackend`](crate::MatrixBackend) and the callable runs
//! [`dispatch_matrix`](crate::dispatch_matrix)).
//!
//! Constraints (the generics are parsed by a small `macro_rules` stripper): the scalar type
//! parameter must be named `T`; an optional single lifetime comes first; type parameters take at
//! most one bound (`T: Scalar`); const generics are supported (`const M: usize`). Invoke at item
//! scope. For anything outside that shape, write the `Kernel` impl by hand.

/// Generate a [`Kernel`](crate::Kernel) or [`MatrixKernel`](crate::MatrixKernel) and a dispatching
/// wrapper from one annotated function. See the [module docs](crate::kernel_macro).
#[macro_export]
macro_rules! kernel {
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident [ $($generics:tt)* ]
        ( $ctx:ident $(, $pn:ident : $pty:ty )* $(,)? ) -> $ret:ty
        $body:block
    ) => {
        $crate::__kernel_strip! {
            generics { $($generics)* } use_acc { } tc_acc { }
            payload {
                meta { $(#[$meta])* } vis { $vis } name { $name } decl { $($generics)* }
                ctx { $ctx } fields { $( $pn : $pty , )* } ret { $ret } body { $body }
                ktrait { $crate::Kernel } bound { $crate::Backend }
                dispatch { $crate::dispatch } dbound { $crate::SimdDispatch }
            }
        }
    };
    (
        $(#[$meta:meta])*
        $vis:vis matrix fn $name:ident [ $($generics:tt)* ]
        ( $ctx:ident $(, $pn:ident : $pty:ty )* $(,)? ) -> $ret:ty
        $body:block
    ) => {
        $crate::__kernel_strip! {
            generics { $($generics)* } use_acc { } tc_acc { }
            payload {
                meta { $(#[$meta])* } vis { $vis } name { $name } decl { $($generics)* }
                ctx { $ctx } fields { $( $pn : $pty , )* } ret { $ret } body { $body }
                ktrait { $crate::MatrixKernel } bound { $crate::MatrixBackend }
                dispatch { $crate::dispatch_matrix } dbound { $crate::MatrixDispatch }
            }
        }
    };
}

/// Strip bounds off the generic list to recover two names-only forms: `use_acc` keeps every name
/// (`<'a, T: Scalar, const M: usize>` → `'a, T, M`) for the `impl … for __Kernel<here>` position;
/// `tc_acc` drops lifetimes (`T, M`) for the constructor turbofish (which rejects lifetimes and is
/// needed so const generics — phantom in the fields — are inferable). CPS over the payload.
#[doc(hidden)]
#[macro_export]
macro_rules! __kernel_strip {
    (generics { } use_acc { $($u:tt)* } tc_acc { $($tc:tt)* } payload { $($p:tt)* }) => {
        $crate::__kernel_emit! { useg { $($u)* } tcg { $($tc)* } $($p)* }
    };
    // const generic (keyword `const` never matches `$:ident`, so these must come first)
    (generics { const $n:ident : $cty:ty } use_acc { $($u:tt)* } tc_acc { $($tc:tt)* } payload { $($p:tt)* }) => {
        $crate::__kernel_strip! { generics { } use_acc { $($u)* $n , } tc_acc { $($tc)* $n , } payload { $($p)* } }
    };
    (generics { const $n:ident : $cty:ty , $($rest:tt)* } use_acc { $($u:tt)* } tc_acc { $($tc:tt)* } payload { $($p:tt)* }) => {
        $crate::__kernel_strip! { generics { $($rest)* } use_acc { $($u)* $n , } tc_acc { $($tc)* $n , } payload { $($p)* } }
    };
    // lifetime (kept in use_acc only)
    (generics { $lt:lifetime } use_acc { $($u:tt)* } tc_acc { $($tc:tt)* } payload { $($p:tt)* }) => {
        $crate::__kernel_strip! { generics { } use_acc { $($u)* $lt , } tc_acc { $($tc)* } payload { $($p)* } }
    };
    (generics { $lt:lifetime , $($rest:tt)* } use_acc { $($u:tt)* } tc_acc { $($tc:tt)* } payload { $($p:tt)* }) => {
        $crate::__kernel_strip! { generics { $($rest)* } use_acc { $($u)* $lt , } tc_acc { $($tc)* } payload { $($p)* } }
    };
    // type param with a single bound
    (generics { $n:ident : $bound:path } use_acc { $($u:tt)* } tc_acc { $($tc:tt)* } payload { $($p:tt)* }) => {
        $crate::__kernel_strip! { generics { } use_acc { $($u)* $n , } tc_acc { $($tc)* $n , } payload { $($p)* } }
    };
    (generics { $n:ident : $bound:path , $($rest:tt)* } use_acc { $($u:tt)* } tc_acc { $($tc:tt)* } payload { $($p:tt)* }) => {
        $crate::__kernel_strip! { generics { $($rest)* } use_acc { $($u)* $n , } tc_acc { $($tc)* $n , } payload { $($p)* } }
    };
    // bare type param
    (generics { $n:ident } use_acc { $($u:tt)* } tc_acc { $($tc:tt)* } payload { $($p:tt)* }) => {
        $crate::__kernel_strip! { generics { } use_acc { $($u)* $n , } tc_acc { $($tc)* $n , } payload { $($p)* } }
    };
    (generics { $n:ident , $($rest:tt)* } use_acc { $($u:tt)* } tc_acc { $($tc:tt)* } payload { $($p:tt)* }) => {
        $crate::__kernel_strip! { generics { $($rest)* } use_acc { $($u)* $n , } tc_acc { $($tc)* $n , } payload { $($p)* } }
    };
}

/// Emit the kernel struct, its trait impl, and the dispatching wrapper. The struct lives in a
/// private module named like the function (modules are a separate namespace from the `fn`), so the
/// generated names never collide with the caller's.
#[doc(hidden)]
#[macro_export]
macro_rules! __kernel_emit {
    (
        useg { $($useg:tt)* } tcg { $($tcg:tt)* }
        meta { $(#[$meta:meta])* } vis { $vis:vis } name { $name:ident } decl { $($decl:tt)* }
        ctx { $ctx:ident } fields { $( $pn:ident : $pty:ty , )* } ret { $ret:ty } body { $body:block }
        ktrait { $($ktrait:tt)* } bound { $($bound:tt)* }
        dispatch { $($dispatch:tt)* } dbound { $($dbound:tt)* }
    ) => {
        mod $name {
            #[allow(unused_imports)]
            use super::*;
            pub(super) struct __Kernel< $($decl)* > {
                $( pub(super) $pn : $pty , )*
            }
            impl< $($decl)* > $($ktrait)*<T> for __Kernel< $($useg)* > {
                type Output = $ret;
                #[inline]
                fn run<__S: $($bound)*<T>>(self, $ctx: $crate::Simd<T, __S>) -> $ret {
                    let __Kernel { $( $pn , )* } = self;
                    $body
                }
            }
        }
        $(#[$meta])*
        #[inline]
        #[allow(clippy::multiple_bound_locations)] // `T: Scalar` (decl) + `T: …Dispatch` (where)
        $vis fn $name < $($decl)* >( $( $pn : $pty , )* ) -> $ret
        where
            T: $($dbound)*,
        {
            // turbofish the type+const generics (lifetimes elided) so const dims that are phantom in
            // the fields are still inferable at construction.
            $($dispatch)*($name::__Kernel::<$($tcg)*> { $( $pn , )* })
        }
    };
}
