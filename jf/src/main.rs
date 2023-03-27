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
    Call(String, Vec<F>),
}

enum FPath {
    Done,
    Single(Context),
    Stream(FStream),
}
impl Iterator for FPath {
    type Item = Context;

    fn next(&mut self) -> Option<Self::Item> {
        let ret: Option<Context>;
        (ret, *self) = match std::mem::replace(self, FPath::Done) {
            FPath::Done => (None, FPath::Done),
            FPath::Single(ctx) => (Some(ctx), FPath::Done),
            FPath::Stream(mut f) => match f.next() {
                Some(ctx) => (Some(ctx), FPath::Stream(f)),
                None => (None, FPath::Done),
            },
        };
        ret
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
        match f {
            F::Literal(lit) => FPath::Single((ctx.0, ctx.1[lit].clone())),
            F::Dot => FPath::Single((ctx.0, ctx.1[&ctx0.1].clone())),
            _ => FPath::Stream(Box::new(
                f.feval(ctx0.clone())
                    .map(move |(_, v)| (ctx.0.clone(), ctx.1[&v].clone())),
            )),
        }
    }

    fn feval(&self, ctx: Context) -> FIter {
        match self {
            // "foo", 1, null, true
            F::Literal(val) => FIter::One((ctx.0, val.clone())),
            // .
            F::Dot => FIter::One(ctx),
            // .[]
            F::Spread => match ctx.1 {
                Value::Array(mut es) => match es.len() {
                    0 => FIter::Zero,
                    1 => FIter::One((ctx.0, es.pop().unwrap())),
                    _ => FIter::Many(Box::new(es.into_iter().map(move |v| (ctx.0.clone(), v)))),
                },
                Value::Object(mut fs) => match fs.len() {
                    0 => FIter::Zero,
                    1 => FIter::One((ctx.0, fs.pop().unwrap().1)),
                    _ => FIter::Many(Box::new(
                        fs.into_iter().map(move |(_, v)| (ctx.0.clone(), v)),
                    )),
                },
                _ => panic!("cannot iterate over {} ({})", ctx.1.kind(), ctx.1),
            },
            // [f1, f2, ..]
            F::Array(fs) => FIter::One((
                ctx.0.clone(),
                Value::Array(
                    fs.iter()
                        .flat_map(|f| f.feval(ctx.clone()))
                        .map(|(_, v)| v)
                        .collect(),
                ),
            )),
            // {}
            F::Object(kfs) if kfs.is_empty() => FIter::One((ctx.0, Value::Object(vec![]))),
            // {"k1": f1, "k2": f2, ..}
            F::Object(kfs) => {
                let mut it = kfs.iter().cloned();
                let hd = it.next().unwrap();
                let mut prod: Box<dyn Iterator<Item = Vec<(String, Value)>>> =
                    match hd.1.feval(ctx.clone()) {
                        FIter::Zero => return FIter::Zero,
                        FIter::One((_, val)) => todo!(),
                        fit => Box::new(
                            hd.1.feval(ctx.clone())
                                .map(move |(_, v)| vec![(hd.0.clone(), v)]),
                        ),
                    };
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
                FIter::Many(fstm)
            }
            // ? weird case
            F::Path(fs) if fs.is_empty() => FIter::Zero,
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
                FIter::Many(fstm)
            }
            // f as $name
            F::Set(name, fval) => {
                let mut it = fval.feval(ctx.clone());
                let (env, val) = ctx;
                let name = name.clone();
                FIter::Many(Box::new(it.map(move |(_, v)| {
                    let mut env = env.clone();
                    env.set(&name, v);
                    (env, val.clone())
                })))
            }
            // $name
            F::Get(name) => {
                let val = ctx.0.get(name).unwrap_or(Value::Null);
                FIter::One((ctx.0, val))
            }
            // ? weird case
            F::Pipe(fs) if fs.is_empty() => FIter::Zero,
            // f1 | f2 | ..
            F::Pipe(fs) => {
                let mut it = fs.iter();
                let mut fstm: FStream = Box::new(it.next().unwrap().feval(ctx));
                FIter::Many(it.fold(fstm, |it, f| {
                    let f = f.clone();
                    Box::new(it.flat_map(move |ctx| f.feval(ctx)))
                }))
            }

            // function defined in wider scope
            F::Call(name, args) => {
                //
                todo!()
            }
        }
    }
}
type FStream = Box<dyn Iterator<Item = Context>>;
enum FIt<I: Iterator> {
    Zero,
    One(I::Item),
    Many(I),
}
type FIter = FIt<FStream>;
impl Iterator for FIter {
    type Item = Context;

    fn next(&mut self) -> Option<Self::Item> {
        let ret: Option<Self::Item>;
        (ret, *self) = match std::mem::replace(self, FIter::Zero) {
            FIter::Zero => (None, FIter::Zero),
            FIter::One(ctx) => (Some(ctx), FIter::Zero),
            FIter::Many(mut fstm) => match fstm.next() {
                Some(ctx) => (Some(ctx), FIter::Many(fstm)),
                None => (None, FIter::Zero),
            },
        };
        ret
    }
}

impl std::ops::BitOr<F> for Value {
    type Output = FIter;

    fn bitor(self, rhs: F) -> Self::Output {
        rhs.feval((Env::new(), self))
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
