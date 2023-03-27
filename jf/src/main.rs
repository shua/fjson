#![allow(unused)]

use std::{
    iter::{once, Once, Peekable},
    mem::MaybeUninit,
    rc::Rc,
    sync::Mutex,
};

#[derive(Debug, Clone)]
enum Value {
    Null,
    Bool(bool),
    Number(i64),
    String(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

macro_rules! impl_vfrom {
    ($($value:ident : $from:ty => $to:expr ,)*) => {$(
    impl From<$from> for Value {
        fn from($value: $from) -> Self { $to } }
    )*};
}
impl_vfrom! {
    b: bool => Value::Bool(b),
    n: i64 => Value::Number(n),
    s: String => Value::String(s),
    s: &'_ str => Value::String(s.to_string()),
    a: Vec<Value> => Value::Array(a),
    a: &'_ [Value] => Value::Array(a.to_vec()),
    kvs: Vec<(String, Value)> => Value::Object(kvs),
    kvs: &'_ [(String, Value)] => Value::Object(kvs.to_vec()),
    kvs: &'_ [(&'_ str, Value)] => Value::Object(kvs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()),
}
impl<const N: usize> From<[Value; N]> for Value {
    fn from(value: [Value; N]) -> Self {
        Value::Array(value.into_iter().collect())
    }
}
impl<const N: usize> From<[(String, Value); N]> for Value {
    fn from(value: [(String, Value); N]) -> Self {
        Value::Object(value.into_iter().collect())
    }
}
impl<T> From<Option<T>> for Value
where
    Value: From<T>,
{
    fn from(value: Option<T>) -> Self {
        value.map(Value::from).unwrap_or(Value::Null)
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Value::*;
        match self {
            Null => write!(f, "null"),
            Bool(b) => write!(f, "{b}"),
            Number(n) => write!(f, "{n}"),
            String(s) => write!(f, "{s:?}"),
            Array(es) if es.is_empty() => write!(f, "[]"),
            Array(es) => {
                write!(f, "[{}", es[0])?;
                for e in &es[1..] {
                    write!(f, ",{e}")?;
                }
                write!(f, "]")
            }
            Object(fs) if fs.is_empty() => write!(f, "{{}}"),
            Object(fs) => {
                write!(f, "{{{:?}:{}", fs[0].0, fs[0].1)?;
                for (k, v) in &fs[1..] {
                    write!(f, ",{k:?}:{v}")?;
                }
                write!(f, "}}")
            }
        }
    }
}

impl Value {
    fn kind(&self) -> &'static str {
        match self {
            Value::Null => "null",
            Value::Bool(_) => "bool",
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        }
    }
}

impl<'s> std::ops::Index<&'s str> for Value {
    type Output = Value;
    fn index(&self, idx: &'s str) -> &Value {
        match self {
            Value::Array(es) => &es[idx.parse::<usize>().unwrap()],
            Value::Object(fs) => fs.iter().find(|(k, _)| k == idx).map(|(_, v)| v).unwrap(),
            _ => panic!("unable to index into {} with string", self.kind()),
        }
    }
}
impl std::ops::Index<usize> for Value {
    type Output = Value;
    fn index(&self, idx: usize) -> &Value {
        match self {
            Value::Array(es) => &es[idx],
            Value::Object(_) => self.index(idx.to_string().as_str()),
            _ => panic!("unable to index into {} with number", self.kind()),
        }
    }
}
impl std::ops::Index<&Value> for Value {
    type Output = Value;
    fn index(&self, idx: &Value) -> &Value {
        match (self, idx) {
            (Value::Array(es), &Value::Number(n)) => &es[usize::try_from(n).unwrap()],
            (Value::Object(fs), &Value::Number(n)) => self.index(n.to_string().as_str()),
            (Value::Object(fs), Value::String(s)) => self.index(s.as_str()),
            _ => panic!("unable to index into {} with {}", self.kind(), idx.kind()),
        }
    }
}

#[derive(Clone)]
enum Env {
    Empty,
    Borrowed(Rc<Env>),
    Owned {
        ext: Rc<Env>,
        int: Vec<(String, Value)>,
    },
}
impl Default for Env {
    fn default() -> Self {
        Env::Empty
    }
}
impl Env {
    fn new() -> Env {
        Default::default()
    }

    fn root() -> Rc<Env> {
        static mut ROOT: MaybeUninit<Rc<Env>> = MaybeUninit::uninit();
        static ROOT_INIT: std::sync::Once = std::sync::Once::new();
        ROOT_INIT.call_once(|| unsafe {
            ROOT.write(Rc::new(Env::Empty));
        });
        unsafe { ROOT.assume_init_ref().clone() }
    }

    fn set(&mut self, key: impl Into<String>, val: impl Into<Value>) {
        let kv = (key.into(), val.into());
        match self {
            Env::Empty => {
                *self = Env::Owned {
                    ext: Env::root(),
                    int: vec![kv],
                }
            }
            Env::Borrowed(ext) => {
                *self = Env::Owned {
                    ext: ext.clone(),
                    int: vec![kv],
                }
            }
            Env::Owned { ext, int } => int.push(kv),
        }
    }
    fn get(&self, key: impl AsRef<str>) -> Option<Value> {
        match self {
            Env::Empty => None,
            Env::Borrowed(ext) => ext.get(key),
            Env::Owned { ext, int } => int
                .iter()
                .find(|(k, v)| k.as_str() == key.as_ref())
                .map(|(_, v)| v.clone())
                .or_else(|| ext.get(key)),
        }
    }
}

type Context = (Env, Value);

trait Filter {
    type Out: Iterator<Item = Context>;

    fn feval(&self, ctx: Context) -> Self::Out;

    fn fbox(mut self) -> DynFilter
    where
        Self: Sized + 'static,
    {
        DynFilter(Rc::new(move |ctx| Box::new(self.feval(ctx))))
    }
}
impl<Out> Filter for fn(Context) -> Out
where
    Out: Iterator<Item = Context>,
{
    type Out = Out;
    fn feval(&self, ctx: Context) -> Out {
        self(ctx)
    }
}

#[derive(Clone)]
struct CloseFilter<F>(F);
impl<F, It> Filter for CloseFilter<F>
where
    F: Fn(Context) -> It,
    It: Iterator<Item = Context>,
{
    type Out = It;
    fn feval(&self, ctx: Context) -> It {
        self.0(ctx)
    }
}

#[derive(Clone)]
struct DynFilter(Rc<dyn Fn(Context) -> Box<dyn Iterator<Item = Context>>>);
impl Filter for DynFilter {
    type Out = Box<dyn Iterator<Item = Context>>;

    fn feval(&self, ctx: Context) -> Self::Out {
        self.0(ctx)
    }
}

fn literal(val: impl Into<Value>) -> impl Filter {
    let val = val.into();
    CloseFilter(move |(env, _)| once((env, val.clone())))
}

// .foo  .[0] .["foo"] .[f]
fn path(mut f: impl Filter) -> impl Filter {
    CloseFilter(move |(env, val): Context| {
        f.feval((env.clone(), val.clone()))
            .map(move |(_, key)| (env.clone(), val[&key].clone()))
    })
}

// .[]
struct Spread(Env, Vec<Value>);
impl Iterator for Spread {
    type Item = Context;
    fn next(&mut self) -> Option<Context> {
        self.1.pop().map(|v| (self.0.clone(), v))
    }
}
fn spread((env, val): Context) -> Spread {
    match val {
        Value::Array(es) => Spread(env, es.into_iter().rev().collect()),
        Value::Object(fs) => Spread(env, fs.into_iter().map(|(_, v)| v).rev().collect()),
        v => panic!("unable to apply .[] to {}", v.kind()),
    }
}
const SPREAD: fn(Context) -> Spread = spread;
const DOT: fn(Context) -> Once<Context> = once;

#[derive(Clone)]
struct MkArray(Vec<DynFilter>);
impl MkArray {
    fn push(&mut self, mut f: impl Filter + 'static) {
        let f = DynFilter(Rc::new(move |ctx| Box::new(f.feval(ctx))));
        self.0.push(f);
    }
    fn v(mut self, f: impl Filter + 'static) -> Self {
        self.push(f);
        self
    }

    fn end(mut self) -> impl Filter {
        CloseFilter(move |(env, val): Context| {
            let mut vs = vec![];
            let mut fs = self.0.clone();
            for fval in fs.iter_mut().rev() {
                vs.extend(fval.0((env.clone(), val.clone())).map(|(_, v)| v));
            }
            std::iter::once((env.clone(), Value::Array(vs)))
        })
    }
}
// [ f, g, ... ]
fn mkarray() -> MkArray {
    MkArray(vec![])
}

#[derive(Clone)]
struct MkObject(Vec<(String, DynFilter)>);
struct MkObjectIterField {
    key: String,
    fval: DynFilter,
    cur: usize,
    next: Peekable<Box<dyn Iterator<Item = Context>>>,
}
impl MkObjectIterField {
    fn new(key: String, fval: DynFilter, ctx: Context) -> Self {
        let next = fval.0(ctx).peekable();
        MkObjectIterField {
            key,
            fval,
            cur: 0,
            next,
        }
    }
}
enum MkObjectIter {
    Done,
    First {
        input: Context,
        fs: Vec<(String, DynFilter)>,
    },
    Rest {
        input: Context,
        fs: Vec<MkObjectIterField>,
    },
}
impl Clone for MkObjectIter {
    fn clone(&self) -> Self {
        match self {
            MkObjectIter::Done => MkObjectIter::Done,
            MkObjectIter::First { input, fs } => MkObjectIter::First {
                input: input.clone(),
                fs: fs.clone(),
            },
            MkObjectIter::Rest { input, fs } => MkObjectIter::Rest {
                input: input.clone(),
                fs: fs
                    .iter()
                    .map(|fld| MkObjectIterField {
                        key: fld.key.clone(),
                        fval: fld.fval.clone(),
                        cur: fld.cur,
                        next: (Box::new(fld.fval.0(input.clone()).skip(fld.cur))
                            as Box<dyn Iterator<Item = Context>>)
                            .peekable(),
                    })
                    .collect(),
            },
        }
    }
}
impl Iterator for MkObjectIter {
    type Item = Context;
    fn next(&mut self) -> Option<Context> {
        match self {
            MkObjectIter::Done => return None,
            &mut MkObjectIter::First { .. } => {
                let (input, fs) = if let MkObjectIter::First { input, fs } =
                    std::mem::replace(self, MkObjectIter::Done)
                {
                    (input, fs)
                } else {
                    unreachable!("guarded by upper match")
                };
                let (ret, mkobjit) = MkObjectIter::first(input, fs)?;
                *self = mkobjit;
                Some(ret)
            }
            &mut MkObjectIter::Rest {
                ref mut input,
                ref mut fs,
            } => {
                let mut should_advance = true;
                let mut flds = vec![];
                for fld in fs.iter_mut() {
                    if !should_advance {
                        let (_, v) = (fld.next.peek())
                            .expect("already checked peekable in previous iteration");
                        flds.push((fld.key.clone(), v.clone()));
                        continue;
                    }
                    fld.next.next();
                    if fld.next.peek().is_some() {
                        should_advance = false;
                    } else {
                        // restart
                        fld.next = fld.fval.feval(input.clone()).peekable();
                    }
                    let (_, v) =
                        (fld.next.peek()).expect("already checked that the fields were non-empty");
                    flds.push((fld.key.clone(), v.clone()));
                }
                if should_advance {
                    // all of the field filters have been iterated through
                    *self = MkObjectIter::Done;
                    return None;
                }
                Some((input.0.clone(), Value::Object(flds)))
            }
        }
    }
}
impl MkObjectIter {
    fn first(ctx: Context, fs: Vec<(String, DynFilter)>) -> Option<(Context, MkObjectIter)> {
        // short-circuit case with no fields, emit single {}
        if fs.is_empty() {
            return Some(((ctx.0, Value::Object(vec![])), MkObjectIter::Done));
        }

        let mut fs: Vec<_> = fs
            .into_iter()
            .map(|(key, fval)| MkObjectIterField::new(key, fval, ctx.clone()))
            .collect();
        if fs.iter_mut().any(|fld| fld.next.peek().is_none()) {
            return None;
        }
        let ret: Vec<_> = fs
            .iter_mut()
            .map(|fld| {
                (
                    fld.key.clone(),
                    fld.next.peek().map(|(_, v)| v).cloned().unwrap(),
                )
            })
            .collect();
        Some((
            (ctx.0.clone(), Value::Object(ret)),
            MkObjectIter::Rest { input: ctx, fs },
        ))
    }
}
impl MkObject {
    fn push(&mut self, key: String, mut f: impl Filter + 'static) {
        let f = DynFilter(Rc::new(move |ctx| Box::new(f.feval(ctx))));
        self.0.push((key, f));
    }
    fn kv(mut self, key: impl Into<String>, mut f: impl Filter + 'static) -> Self {
        self.push(key.into(), f);
        self
    }
    fn end(mut self) -> impl Filter {
        CloseFilter(move |(env, val)| MkObjectIter::First {
            input: (env, val),
            fs: self.0.clone(),
        })
    }
}
fn mkobject() -> MkObject {
    MkObject(vec![])
}

fn length((env, val): Context) -> impl Iterator<Item = Context> {
    match val {
        Value::String(s) => once((env, Value::Number(i64::try_from(s.len()).unwrap()))),
        Value::Array(es) => once((env, Value::Number(i64::try_from(es.len()).unwrap()))),
        _ => panic!("unable to take length of {}", val.kind()),
    }
}

fn keys((env, val): Context) -> impl Iterator<Item = Context> {
    match val {
        Value::Object(fs) => once((
            env,
            Value::Array(fs.into_iter().map(|(k, _)| Value::String(k)).collect()),
        )),
        _ => panic!("unable to take keys of {}", val.kind()),
    }
}

struct Bind<I>(I, Context, String);
impl<I> std::iter::Iterator for Bind<I>
where
    I: std::iter::Iterator<Item = Context>,
{
    type Item = Context;

    fn next(&mut self) -> Option<Self::Item> {
        let it = &mut self.0;
        let name = self.2.as_str();
        it.next().map(|(_, v)| {
            let (mut env, val) = self.1.clone();
            env.set(name, v);
            (env, val)
        })
    }
}
fn setenv(name: impl AsRef<str>, fval: impl Filter) -> impl Filter {
    CloseFilter(move |ctx: Context| Bind(fval.feval(ctx.clone()), ctx, name.as_ref().to_string()))
}
fn getenv(name: impl AsRef<str>) -> impl Filter {
    CloseFilter(move |(env, _): Context| {
        let val = env.get(&name).unwrap_or(Value::Null);
        once((env, val))
    })
}

impl<F: Filter + 'static> std::ops::BitOr<F> for DynFilter {
    type Output = DynFilter;

    fn bitor(mut self, mut rhs: F) -> Self::Output {
        let rhs = Rc::new(rhs);
        DynFilter(Rc::new(move |ctx| {
            let mut it = self.0(ctx);
            let rhs = rhs.clone();
            Box::new(it.flat_map(move |ctx| rhs.feval(ctx)))
        }))
    }
}
impl<F, Cf, Cout> std::ops::BitOr<F> for CloseFilter<Cf>
where
    F: Filter + 'static,
    Cf: Fn(Context) -> Cout + 'static,
    Cout: Iterator<Item = Context> + 'static,
{
    type Output = DynFilter;
    fn bitor(mut self, mut rhs: F) -> Self::Output {
        let rhs = Rc::new(rhs);
        DynFilter(Rc::new(move |ctx| {
            let mut it = self.0(ctx);
            let rhs = rhs.clone();
            Box::new(it.flat_map(move |ctx| rhs.feval(ctx)))
        }))
    }
}
impl<F> std::ops::BitOr<F> for Value
where
    F: Filter,
{
    type Output = F::Out;

    fn bitor(self, rhs: F) -> Self::Output {
        rhs.feval((Env::new(), self))
    }
}

macro_rules! json {
    ([ $($val:tt),* ]) => { Value::Array(vec![ $(json!($val)),* ]) };
    ({ $($key:literal : $val:tt),* }) => {
        Value::Object(vec![$( (
            ($key).to_string(),
            Value::from(json!($val))

        ) ),*])
    };
    (null) => { Value::Null };
    ($val:expr) => { Value::from($val) };
}

#[derive(Clone)]
enum F {
    Literal(Value),
    Dot,
    Spread,
    Array(Vec<F>),
    Object(Vec<(String, F)>),
    Path(Vec<F>),
    Set(String, Box<F>),
    Get(String),
    Pipe(Vec<F>),
}
enum FPath {
    Done,
    Single(Context),
    Stream(FStream),
}
impl Iterator for FPath {
    type Item = Context;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            FPath::Done => None,
            FPath::Single(_) => {
                let ctx = match std::mem::replace(self, FPath::Done) {
                    FPath::Single(ctx) => ctx,
                    _ => unreachable!("outer match"),
                };
                Some(ctx)
            }
            FPath::Stream(f) => match f.next() {
                Some(ctx) => Some(ctx),
                None => {
                    *self = FPath::Done;
                    None
                }
            },
        }
    }
}
impl F {
    fn fpath(ctx: Context, ctx0: &Context, f: &F) -> FPath {
        // mostly translating
        //    .[.|f1][.|f2].l3
        // to . as v0 | .[v0 | f1] | .[v0 | f2] | .[l3]
        // recursively indexing, but have to keep track of the value that
        // is in pipe-level scope (here shown as v0).
        // simply breaking up to
        //    .[.|f1] | .[.|f2]
        // doesn't work because .|f2 becomes a different value
        // We special case for literals (.foo/.["foo"]/.[1]) or .[.]
        // because we don't need to make much of a
        match f {
            F::Literal(lit) => FPath::Single((ctx.0, ctx.1[lit].clone())),
            F::Dot => FPath::Single((ctx.0, ctx.1[&ctx0.1].clone())),
            _ => FPath::Stream(Box::new(
                f.feval(ctx0.clone())
                    .map(move |(_, v)| (ctx.0.clone(), ctx.1[&v].clone())),
            )),
        }
    }
}
type FStream = Box<dyn Iterator<Item = Context>>;
enum FIter {
    Done,
    Dot(Context),
    Spread(Context),
    Stream(FStream),
}
impl Iterator for FIter {
    type Item = Context;

    fn next(&mut self) -> Option<Self::Item> {
        let ret: Option<Self::Item>;
        (ret, *self) = match std::mem::replace(self, FIter::Done) {
            FIter::Done => (None, FIter::Done),
            FIter::Dot(ctx) => (Some(ctx), FIter::Done),

            FIter::Spread((env, Value::Array(mut es))) => match es.len() {
                0 | 1 => (es.pop().map(|e| (env, e)), FIter::Done),
                _ => (
                    Some((env.clone(), es.pop().unwrap())),
                    FIter::Spread((env, Value::Array(es))),
                ),
            },
            FIter::Spread((env, Value::Object(mut fs))) => match fs.len() {
                0 | 1 => (fs.pop().map(|(_, v)| (env, v)), FIter::Done),
                _ => (
                    Some((env.clone(), fs.pop().unwrap().1)),
                    FIter::Spread((env, Value::Object(fs))),
                ),
            },
            FIter::Spread((_env, val)) => panic!("cannot iterate over {} ({})", val.kind(), val),

            FIter::Stream(mut fstm) => match fstm.next() {
                Some(ctx) => (Some(ctx), FIter::Stream(fstm)),
                None => (None, FIter::Done),
            },
        };
        ret
    }
}
impl Filter for F {
    type Out = FIter;

    fn feval(&self, ctx: Context) -> Self::Out {
        match self {
            // "foo", 1, null, true
            F::Literal(val) => FIter::Dot((ctx.0, val.clone())),
            // .
            F::Dot => FIter::Dot(ctx),
            // .[]
            F::Spread => FIter::Spread(ctx),
            // [f1, f2, ..]
            F::Array(fs) => FIter::Dot((
                ctx.0.clone(),
                Value::Array(
                    fs.iter()
                        .flat_map(|f| f.feval(ctx.clone()))
                        .map(|(_, v)| v)
                        .collect(),
                ),
            )),
            // {}
            F::Object(kfs) if kfs.is_empty() => FIter::Dot((ctx.0, Value::Object(vec![]))),
            // {"k1": f1, "k2": f2, ..}
            F::Object(kfs) => {
                let mut it = kfs.iter().cloned();
                let hd = it.next().unwrap();
                let mut prod: Box<dyn Iterator<Item = Vec<(String, Value)>>> = Box::new(
                    hd.1.feval(ctx.clone())
                        .map(move |(_, v)| vec![(hd.0.clone(), v)]),
                );
                for (k, f) in it {
                    let ctx = ctx.clone();
                    prod = Box::new(prod.flat_map(move |kvs| {
                        let k = k.clone();
                        let next = f.feval(ctx.clone()).map(move |(_, v)| (k.clone(), v));
                        next.map(move |kv| {
                            let mut kvs = kvs.clone();
                            kvs.push(kv);
                            kvs
                        })
                    }))
                }
                let fstm = Box::new(prod.map(move |kvs| (ctx.0.clone(), Value::Object(kvs))));
                FIter::Stream(fstm)
            }
            // ? weird case
            F::Path(fs) if fs.is_empty() => FIter::Done,
            // .[f1][f2] ..
            F::Path(fs) => {
                let mut it = fs.iter();
                let ctx0 = Rc::new(ctx.clone());
                let first = it.next().unwrap();
                let first: FStream = Box::new(F::fpath(ctx.clone(), ctx0.clone().as_ref(), first));
                let fstm = it.fold(first, |it, f| {
                    let ctx0 = ctx0.clone();
                    let f = f.clone();
                    Box::new(it.flat_map(move |ctx| F::fpath(ctx, ctx0.clone().as_ref(), &f)))
                });
                FIter::Stream(fstm)
            }
            // f as $name
            F::Set(name, fval) => {
                let mut it = fval.feval(ctx.clone());
                let (env, val) = ctx;
                let name = name.clone();
                FIter::Stream(Box::new(it.map(move |(_, v)| {
                    let mut env = env.clone();
                    env.set(&name, v);
                    (env, val.clone())
                })))
            }
            // $name
            F::Get(name) => {
                let val = ctx.0.get(name).unwrap_or(Value::Null);
                FIter::Dot((ctx.0, val))
            }
            // ? weird case
            F::Pipe(fs) if fs.is_empty() => FIter::Done,
            // f1 | f2 | ..
            F::Pipe(fs) => {
                let mut it = fs.iter();
                let mut fstm: FStream = Box::new(it.next().unwrap().feval(ctx));
                FIter::Stream(it.fold(fstm, |it, f| {
                    let f = f.clone();
                    Box::new(it.flat_map(move |ctx| f.feval(ctx)))
                }))
            }
        }
    }
}

// For the most part, jf exprs are tt's, eg [...], {...}, (...), ., "foo", 4, true, null
// however, there are 4 exceptions:
//   paths: .[f1][f2]...
//   pipes: f1 | f2 | ...
//   sets:  f as @name
//   gets:  @name
// for any rules that are recursive, we need a simple case that just matches a single tt
// and 4 exceptional cases that match paths, pipes, sets and gets.
// Since sets are left recursive, we can match paths pipes and gets, with an optional
// "as @name" tacked to the end. This excludes things like "f as @foo as @bar".
// If you want that, you'll have to wrap the inner with parens: (f as @foo) as @bar.
//
// Since pipes, arrays and objects can have lists including f's, we'll have to implement
// those as token-munchers:
//   pipes:   f1 | f2 | ...
//   arrays:  [ f1, f2, ... ]
//   objects: { k1: f1, k2: f2, ... }
// Pipes are not tt's while arrays and objects are, so the token munchers for arrays
// and objects will have to handle | and , delimiters, alternatively pushing a filter
// onto the current element, or pushing the current element onto the array/object stack.
macro_rules! fjson {
    (($($t:tt)*)) => { fjson!($($t)*) };

    (.) => { F::Dot };
    (.[]) => { F::Spread };
    (.$([$($f:tt)*])*) => { F::Path(vec![$( fjson!($($f)*) ),*]) };
    (null) => { F::Literal(Value::Null) };

    // get/set
    (@$name:ident) => { F::Get(stringify!($name).to_string()) };
    ($f:tt as @ $name:ident) => { F::Set(stringify!($name).to_string(), Box::new(fjson!($f))) };
    // <path>: '.' ('[' TT* ']')*
    (.$([$($f:tt)*])* as @ $name:ident) => { F::Set(stringify!($name).to_string(), Box::new(fjson!(.$([$($f)*])*)) ) };
    (@$key:ident as @ $name:ident) => { F::Set(stringify!($name).to_string(), Box::new(fjson!(@ $key)) ) };

    // <expr>: (TT / <path> / '@'IDENT) ('as' '@'IDENT)?

    // array simple cases
    ([]) => { F::Array(vec![]) };
    ([$($f:tt),* $(,)?]) => { F::Array(vec![$( fjson!($f) ),*]) };
    // array token muncher
    ([$f:tt $(as @$name:ident)? $(| $($pr:tt)*)? $(, $($er:tt)*)?]) => {
        fjson!(@@array [] (($f $(as @$name)?)) $(| $($pr)*)? $(, $($er)*)?)
    };
    ([.$([$($f:tt)*])* $(as @$name:ident)? $(| $($pr:tt)*)? $(, $($er:tt)*)?]) => {
        fjson!(@@array [] ((.$([$($f)*])* $(as @$name)?)) $(| $($pr)*)? $(, $($er)*)?)
    };
    ([@$name:ident $(as @$key:ident)? $(| $($pr:tt)*)? $(, $($er:tt)*)?]) => {
        fjson!(@@array [] ((@$name $(as @$key)?)) $(| $($pr)*)? $(, $($er)*)?)
    };
    // <a-push-elem>: '@@array' <a-elem> <a-pipe> ',' <expr> <a-tail>
    // <a-push-pipe>: '@@array' <a-elem> <a-pipe> '|' <expr> <a-tail>
    // <a-elem>: '[' ']' / '[' EXPR (',' EXPR)* ']'
    // <a-pipe>: '(' ')' / '(' TT ('|' TT)* ')'
    // <a-tail>: ('|' TT*) / (',' TT*)
    (@@array [$($es:expr),*] ($($ps:tt)|*)) => { F::Array(vec![ $($es,)* fjson!($($ps)|*) ]) };
    (@@array [$($es:expr),*] ($($ps:tt)|*) , $f:tt $(as @$name:ident)? $(| $($pr:tt)*)? $(, $($er:tt)*)? ) => {
        fjson!(@@array [$($es,)* fjson!($($ps)|*)] (($f $(as $name)?)) $(| $($pr)*)? $(, $($er)*)? )
    };
    (@@array [$($es:expr),*] ($($ps:tt)|*) | $f:tt $(as @$name:ident)? $(| $($pr:tt)*)? $(, $($er:tt)*)? ) => {
        fjson!(@@array [$($es),*] ($($ps|)* ($f $(as $name)?)) $(| $($pr)*)? $(, $($er)*)? )
    };
    (@@array [$($es:expr),*] ($($ps:tt)|*) , .$([$($f:tt)*])* $(as @$name:ident)? $(| $($pr:tt)*)? $(, $($er:tt)*)?) => {
        fjson!(@@array [$($es,)* fjson!($($ps)|*)] ((.$([$($f)*])* $(as @$name)?)) $(| $($pr)*)? $(, $($er)*)? )
    };
    (@@array [$($es:expr),*] ($($ps:tt)|*) | .$([$($f:tt)*])* $(as @$name:ident)? $(| $($pr:tt)*)? $(, $($er:tt)*)?) => {
        fjson!(@@array [$($es),*] ($($ps|)* ((.$([$($f)*])* $(as @$name)?))) $(| $($pr)*)? $(, $($er)*)?)
    };
    (@@array [$($es:expr),*] ($($ps:tt)|*) , @$name:ident $(as @$key:ident)? $(| $($pr:tt)*)? $(, $($er:tt)*)?) => {
        fjson!(@@array [$($es,)* fjson!($($ps)|*)] ((@$name $(as @$key)?)) $(| $($pr)*)? $(, $($er)*)?)
    };
    (@@array [$($es:expr),*] ($($ps:tt)|*) | @$name:ident $(as @$key:ident)? $(| $($pr:tt)*)? $(, $($er:tt)*)?) => {
        fjson!(@@array [$($es),*] ($($ps|)* ((@$name $(as @$key)?))) $(| $($pr)*)? $(, $($er)*)?)
    };

    // object simple case
    ({}) => { F::Object(vec![]) };
    ({$($k:literal : $fv:tt),* $(,)?}) => { F::Object(vec![$( ($k.to_string(), fjson!($fv)) ),*]) };
    // object token-muncher
    ({$k:literal : $f:tt $(as @$name:ident)? $(| $($pr:tt)*)? $(, $($er:tt)*)?}) => {
        fjson!(@@object [] $k (($f $(as @$name)?)) $(| $($pr)*)? $(, $($er)*)?)
    };
    ({$k:literal : .$([$($f:tt)*])* $(as @$name:ident)? $(| $($pr:tt)*)? $(, $($er:tt)*)?}) => {
        fjson!(@@object [] $k ((.$([$($f)*])* $(as @$name)?)) $(| $($pr)*)? $(, $($er)*)?)
    };
    ({$k:literal : @$name:ident $(as @$key:ident)? $(| $($pr:tt)*)? $(, $($er:tt)*)?}) => {
        fjson!(@@object [] $k ((@$name $(as @$key)?)) $(| $($pr)*)? $(, $($er)*)?)
    };
    // <o-push-fld>:  '@@object' <o-field> LIT <o-pipe> ',' LIT ':' <expr> <o-tail>
    // <o-push-pipe>: '@@object' <o-field> LIT <o-pipe> '|' <expr> <o-tail>
    // <o-elem>: '[' ']' / '[' EXPR (',' EXPR)* ']'
    // <o-pipe>: '(' ')' / '(' TT ('|' TT)* ')'
    // <o-tail>: ('|' TT*) / (',' TT*)
    (@@object [$($es:expr),*] $k:literal ($($ps:tt)|*)) => { F::Object(vec![$($es,)* ($k.to_string(), fjson!($($ps)|*))]) };
    (@@object [$($es:expr),*] $k:literal ($($ps:tt)|*) , $k2:literal : $f:tt $(as @$name:ident)? $(| $($pr:tt)*)? $(, $($er:tt)*)? ) => {
        fjson!(@@object [$($es,)* ($k.to_string(), fjson!($($ps)|*))] $k2 (($f $(as $name)?)) $(| $($pr)*)? $(, $($er)*)? )
    };
    (@@object [$($es:expr),*] $k:literal ($($ps:tt)|*) | $f:tt $(as @$name:ident)? $(| $($pr:tt)*)? $(, $($er:tt)*)? ) => {
        fjson!(@@object [$($es),*] $k ($($ps|)* ($f $(as $name)?)) $(| $($pr)*)? $(, $($er)*)? )
    };
    (@@object [$($es:expr),*] $k:literal ($($ps:tt)|*) , $k2:literal : .$([$($f:tt)*])* $(as @$name:ident)? $(| $($pr:tt)*)? $(, $($er:tt)*)?) => {
        fjson!(@@object [$($es,)* ($k.to_string(), fjson!($($ps)|*))] $k2 ((.$([$($f)*])* $(as @$name)?)) $(| $($pr)*)? $(, $($er)*)? )
    };
    (@@object [$($es:expr),*] $k:literal ($($ps:tt)|*) | .$([$($f:tt)*])* $(as @$name:ident)? $(| $($pr:tt)*)? $(, $($er:tt)*)?) => {
        fjson!(@@object [$($es),*] $k ($($ps|)* ((.$([$($f)*])* $(as @$name)?))) $(| $($pr)*)? $(, $($er)*)?)
    };
    (@@object [$($es:expr),*] $k:literal ($($ps:tt)|*) , $k2:literal : @$name:ident $(as @$key:ident)? $(| $($pr:tt)*)? $(, $($er:tt)*)?) => {
        fjson!(@@object [$($es,)* ($k.to_string(), fjson!($($ps)|*))] $k2 ((@$name $(as @$key)?)) $(| $($pr)*)? $(, $($er)*)?)
    };
    (@@object [$($es:expr),*] $k:literal ($($ps:tt)|*) | @$name:ident $(as @$key:ident)? $(| $($pr:tt)*)? $(, $($er:tt)*)?) => {
        fjson!(@@object [$($es),*] $k ($($ps|)* ((@$name $(as @$key)?))) $(| $($pr)*)? $(, $($er)*)?)
    };

    // pipes simple case
    ($f:tt | $($g:tt)|*) => { F::Pipe(vec![ fjson!($f), $(fjson!($g)),*]) };
    // pipes token muncher
    ($f:tt $(as @$name:ident)? | $($rest:tt)*) => {
        fjson!(@@pipe [fjson!($f $(as @$name)?)] $($rest)*)
    };
    (.$([$($f:tt)*])* $(as @$name:ident)? | $($rest:tt)*) => {
        fjson!(@@pipe [fjson!(.$([$($f)*])* $(as @$name)?)] $($rest)*)
    };
    (@$name:ident $(as @$key:ident)? | $($rest:tt)*) => {
        fjson!(@@pipe [fjson!(@$name $(as @$key)?)] $($rest)*)
    };
    (@@pipe [$($f:expr),*]) => { F::Pipe(vec![ $($f),* ]) };
    (@@pipe [$($f:expr),*] $g:tt $(as @$name:ident)? $(| $($rest:tt)*)?) => {
        fjson!(@@pipe [$($f,)* fjson!($g $(as @$name)?)] $($($rest)*)?)
    };
    (@@pipe [$($f:expr),*] .$([$($g:tt)*])*  $(as @$name:ident)? $(| $($rest:tt)*)?) => {
        fjson!(@@pipe [$($f,)* fjson!(.$([$($g)*])* $(as @$name)?)] $($($rest)*)?)
    };
    (@@pipe [$($f:expr),*] @$name:ident $(| $($rest:tt)*)?) => {
        fjson!(@@pipe [$($f,)* fjson!(@$name)] $($($rest)*)?)
    };

    ($val:expr) => { F::Literal(Value::from($val)) };
}

fn main() {
    let obj = json!({"a": [1,2,{},null], "b": true});
    let f: F = fjson!(
        {"a": [1,2,{},null], "b": true}
        | .["b"] as @foo
        | [ .["a"] | .[] | . ]
        | {"vals": ., "val": .[], "foo": @foo}
    );
    for (_, v) in obj | f {
        println!("{v}");
    }
}
