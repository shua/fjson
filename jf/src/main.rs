#![allow(unused)]

use std::{fmt::Display, io::Read, rc::Rc};

macro_rules! errexit {
    ($fmt:literal $(, $arg:expr)*) => { errexit!($fmt $(, $arg)* ; 1) };
    ($fmt:literal $(, $arg:expr)* ; $status:expr) => {{
        eprintln!($fmt $(, $arg)*);
        std::process::exit($status);
    }};
}

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

impl Display for Value {
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

    fn get<'v, I: VIndex>(&'v self, idx: I) -> Result<I::Out<'v>, String> {
        idx.index_in(self)
    }
}

trait VIndex {
    type Out<'v>;
    fn index_in<'v>(self, val: &'v Value) -> Result<Self::Out<'v>, String>;
}
impl<'s> VIndex for &'s str {
    type Out<'v> = &'v Value;
    fn index_in(self, val: &Value) -> Result<&Value, String> {
        match val {
            Value::Null => Ok(val),
            Value::Object(fs) => Ok(fs
                .iter()
                .find_map(|(k, v)| (k == self).then(|| v))
                .unwrap_or(&Value::Null)),
            _ => Err(format!("Cannot index {} with string", val.kind())),
        }
    }
}
impl VIndex for i64 {
    type Out<'v> = &'v Value;
    fn index_in(self, val: &Value) -> Result<&Value, String> {
        match val {
            Value::Null => Ok(val),
            Value::Array(es) => {
                let v: Option<&Value>;
                if self >= 0 {
                    v = es.get(usize::try_from(self).unwrap());
                } else {
                    let nidx = usize::try_from(-self).unwrap();
                    if nidx < es.len() {
                        v = es.get(es.len() - nidx);
                    } else {
                        v = None;
                    }
                }
                Ok(v.unwrap_or(&Value::Null))
            }
            _ => Err(format!("Cannot index {} with number", val.kind())),
        }
    }
}
impl<'s> VIndex for &'s Value {
    type Out<'v> = &'v Value;
    fn index_in(self, val: &Value) -> Result<&Value, String> {
        match self {
            Value::String(s) => s.index_in(val),
            Value::Number(n) => (*n).index_in(val),
            _ => Err(format!("Cannot index {} with {}", val.kind(), self.kind())),
        }
    }
}
impl VIndex for std::ops::Range<Option<i64>> {
    type Out<'v> = Value;
    fn index_in(self, val: &Value) -> Result<Value, String> {
        match val {
            Value::Null => Ok(Value::Null),
            Value::Array(es) => {
                let max = i64::try_from(es.len()).unwrap();
                let mut nm = [self.start.unwrap_or(0), self.end.unwrap_or(max)];
                println!("range: {self:?}");
                for n in &mut nm {
                    // -n means "n from the end"
                    if *n < 0 {
                        *n += i64::try_from(es.len()).unwrap();
                    }

                    // bound between 0..len
                    *n = (*n).clamp(0, max);
                }
                let idx = (usize::try_from(nm[0]).unwrap()..usize::try_from(nm[1]).unwrap());
                println!("range: {idx:?}");
                Ok(Value::Array(es[idx].to_vec()))
            }
            _ => Err(format!("Cannot index {} with number:number", val.kind())),
        }
    }
}
impl<'idx> VIndex for std::ops::Range<Option<&'idx Value>> {
    type Out<'v> = Value;

    fn index_in<'v>(self, val: &'v Value) -> Result<Self::Out<'v>, String> {
        match (self.start, self.end) {
            (None, None) => Ok(val.clone()),
            (None, Some(Value::Number(m))) => (None..Some(*m)).index_in(val),
            (Some(Value::Number(n)), None) => (Some(*n)..None).index_in(val),
            (Some(Value::Number(n)), Some(Value::Number(m))) => (Some(*n)..Some(*m)).index_in(val),

            (start, end) => Err(format!(
                "Cannot index {} with {}:{}",
                val.kind(),
                start.map(Value::kind).unwrap_or(""),
                end.map(Value::kind).unwrap_or("")
            )),
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
        use std::{mem::MaybeUninit, sync::Once};
        static mut ROOT: MaybeUninit<Rc<Env>> = MaybeUninit::uninit();
        static ROOT_INIT: Once = Once::new();
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

#[derive(Debug, Clone)]
enum FPath {
    Index(F),
    Range(Option<F>, Option<F>),
    Spread,
    Field(String),
}

impl FPath {
    fn normalize(&mut self) {
        // todo
    }
}

#[derive(Debug, Clone)]
enum F {
    Literal(Value),
    Dot,
    Spread,
    Array(Vec<F>),
    Object(Vec<(String, F)>),
    Path(Vec<(FPath, bool)>),
    Set(String, Box<F>),
    Get(String),
    Pipe(Vec<F>),
    Call(String, Vec<F>),
}

impl F {
    fn fpath(
        (env, val): Context,
        (env0, val0): &Context,
        &(ref p, swallow): &(FPath, bool),
    ) -> FIter {
        // mostly translating
        //    .[.|f1][.|f2].l3
        // to . as v0 | .[v0 | f1] | .[v0 | f2] | .[l3]
        // recursively indexing, but have to keep track of the value that
        // is in pipe-level scope (here shown as v0).
        // simply breaking up to
        //    .[.|f1] | .[.|f2]
        // doesn't work because .|f2 becomes a different value
        // We special case for literals (.foo/.["foo"]/.[1]) or .[.]
        fn swallow_err<E: Display>(swallow: bool) -> impl Fn(E) -> FIter {
            if swallow {
                |_| FIter::Zero
            } else {
                |err| errexit!("error: {err}")
            }
        }
        let with_env = move |res: Result<Value, String>, env| {
            res.map_or_else(swallow_err(swallow), |v| FIter::One((env, v)))
        };
        let cloned_with_env = move |res: Result<&Value, String>, env| {
            res.map(|v| v.clone())
                .map_or_else(swallow_err(swallow), |v| FIter::One((env, v)))
        };
        match p {
            FPath::Index(F::Literal(v)) => cloned_with_env(val.get(v), env),
            FPath::Index(F::Dot) => cloned_with_env(val.get(val0), env),
            FPath::Index(f) => f
                .eval((env0.clone(), val0.clone()))
                .flatter_map(move |(_, v)| cloned_with_env(val.get(&v).clone(), env.clone())),
            FPath::Range(None, None) => {
                with_env(val.get::<std::ops::Range<Option<i64>>>(None..None), env)
            }
            FPath::Range(Some(f), None) => f
                .eval((env0.clone(), val0.clone()))
                .flatter_map(move |(_, v)| with_env(val.get(Some(&v)..None), env.clone())),
            FPath::Range(None, Some(g)) => g
                .eval((env0.clone(), val0.clone()))
                .flatter_map(move |(_, v)| with_env(val.get(None..Some(&v)), env.clone())),
            FPath::Range(Some(f), Some(g)) => match f.eval((env0.clone(), val0.clone())) {
                FIt::Zero => FIter::Zero,
                FIt::One((_, v)) => g
                    .eval((env0.clone(), val0.clone()))
                    .flatter_map(move |(_, w)| with_env(val.get(Some(&v)..Some(&w)), env.clone())),
                FIt::Many(fs) => match g.eval((env0.clone(), val0.clone())) {
                    FIt::Zero => FIter::Zero,
                    FIt::One((_, w)) => FIter::Many(Box::new(fs.flat_map(move |(_, v)| {
                        with_env(val.get(Some(&v)..Some(&w)), env.clone())
                    }))),
                    FIt::Many(_) => {
                        let g = g.clone();
                        let ctx0 = (env0.clone(), val0.clone());
                        FIter::Many(Box::new(fs.flat_map(move |(_, v)| {
                            let gs = g.eval(ctx0.clone());
                            let env = env.clone();
                            let val = val.clone();
                            gs.flatter_map(move |(_, w)| {
                                with_env(val.clone().get(Some(&v)..Some(&w)), env.clone())
                            })
                        })))
                    }
                },
            },
            FPath::Spread => F::Spread.eval((env, val)),
            FPath::Field(name) => cloned_with_env(val.get(name.as_str()), env),
        }
    }

    fn normalize(&mut self) {
        match self {
            F::Literal(_) | F::Dot | F::Spread | F::Get(_) => {}
            F::Array(es) => {
                let mut values = true;
                for e in es.iter_mut() {
                    e.normalize();
                    values &= matches!(*e, F::Literal(_));
                }

                if values {
                    *self = F::Literal(Value::Array(
                        std::mem::take(es)
                            .into_iter()
                            .map(|f| match f {
                                F::Literal(v) => v,
                                _ => unreachable!(),
                            })
                            .collect(),
                    ));
                }
            }
            F::Object(kvs) => {
                let mut values = true;
                for (k, f) in kvs.iter_mut() {
                    f.normalize();
                    values &= matches!(*f, F::Literal(_));
                }

                if values {
                    *self = F::Literal(Value::Object(
                        std::mem::take(kvs)
                            .into_iter()
                            .map(|(k, f)| match f {
                                F::Literal(v) => (k, v),
                                _ => unreachable!(),
                            })
                            .collect(),
                    ));
                }
            }
            F::Path(ps) if ps.is_empty() => *self = F::Dot,
            F::Path(ps) => ps.iter_mut().map(|(p, _)| p.normalize()).collect(),
            F::Set(_, f) => f.normalize(),
            F::Pipe(ref mut fs) if fs.len() == 1 => {
                let mut f = fs.pop().unwrap();
                f.normalize();
                *self = f;
            }
            F::Pipe(fs) => fs.iter_mut().map(F::normalize).collect(),
            F::Call(_, args) => args.iter_mut().map(F::normalize).collect(),
        }
    }

    fn eval(&self, ctx: Context) -> FIter {
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
                _ => errexit!("cannot iterate over {} ({})", ctx.1.kind(), ctx.1),
            },
            // [f1, f2, ..]
            F::Array(fs) => FIter::One((
                ctx.0.clone(),
                Value::Array(
                    fs.iter()
                        .flat_map(|f| f.eval(ctx.clone()))
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
                let mut prod: FIt<Box<dyn Iterator<Item = Vec<(String, Value)>>>> =
                    match hd.1.eval(ctx.clone()) {
                        FIter::Zero => return FIter::Zero,
                        FIter::One((_, val)) => FIt::One(vec![(hd.0, val)]),
                        fit => FIt::Many(Box::new(
                            hd.1.eval(ctx.clone())
                                .map(move |(_, v)| vec![(hd.0.clone(), v)]),
                        )),
                    };
                for (k, f) in it {
                    let ctx = ctx.clone();
                    prod = match (prod, f.eval(ctx.clone())) {
                        (FIt::Zero, _) | (_, FIt::Zero) => return FIter::Zero,
                        // try to minimize mapping and boxing for simpler objects
                        (FIt::One(mut kvs), FIt::One((_, v))) => {
                            kvs.push((k, v));
                            FIt::One(kvs)
                        }
                        (kvs, ctxs) => FIt::Many(Box::new(kvs.flat_map(move |kvs| {
                            let k = k.clone();
                            let next = f.eval(ctx.clone()).map(move |(_, v)| (k.clone(), v));
                            next.map(move |kv| {
                                let mut kvs = kvs.clone();
                                kvs.push(kv);
                                kvs
                            })
                        }))),
                    }
                }

                match prod {
                    FIt::Zero => FIter::Zero,
                    FIt::One(kvs) => FIter::One((ctx.0, Value::Object(kvs))),
                    FIt::Many(it) => FIter::Many(Box::new(
                        it.map(move |kvs| (ctx.0.clone(), Value::Object(kvs))),
                    )),
                }
            }
            // weird case, I guess it's just '.' though?
            F::Path(fs) if fs.is_empty() => FIter::One(ctx),
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
                let mut it = fval.eval(ctx.clone());
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
                let mut fstm: FStream = Box::new(it.next().unwrap().eval(ctx));
                FIter::Many(it.fold(fstm, |it, f| {
                    let f = f.clone();
                    Box::new(it.flat_map(move |ctx| f.eval(ctx)))
                }))
            }

            // function defined in wider scope
            F::Call(name, args) => {
                todo!()
            }
        }
    }
}

struct FPathDisplay<'p>(&'p [(FPath, bool)]);
impl<'p> Display for FPathDisplay<'p> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut last_bracket = false;
        for (p, s) in self.0 {
            if matches!(p, FPath::Field(_)) || !last_bracket {
                write!(f, ".")?;
            }
            last_bracket = true;
            match p {
                FPath::Index(g) => write!(f, "[{g}]"),
                FPath::Range(None, None) => write!(f, "[:]"),
                FPath::Range(Some(g), None) => write!(f, "[{g}:]"),
                FPath::Range(None, Some(h)) => write!(f, "[:{h}]"),
                FPath::Range(Some(g), Some(h)) => write!(f, "[{g}:{h}]"),
                FPath::Spread => write!(f, "[]"),
                FPath::Field(name) => {
                    last_bracket = false;
                    write!(f, "{name}")
                }
            }?;
            if *s {
                write!(f, "?")?;
            }
        }
        Ok(())
    }
}

impl Display for F {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            F::Literal(v) => Display::fmt(v, f),
            F::Dot => Display::fmt(".", f),
            F::Spread => Display::fmt(".[]", f),
            F::Array(es) if es.is_empty() => write!(f, "[]"),
            F::Array(es) => {
                write!(f, "[{}", es[0])?;
                for e in &es[1..] {
                    write!(f, ",{e}")?;
                }
                write!(f, "]")
            }
            F::Object(fs) if fs.is_empty() => write!(f, "{{}}"),
            F::Object(fs) => {
                write!(f, "{{{:?}: {}", fs[0].0, fs[0].1)?;
                for (k, v) in &fs[1..] {
                    write!(f, ", {k:?}: {v}")?;
                }
                write!(f, "}}")
            }
            F::Path(ps) => write!(f, "{}", FPathDisplay(ps)),
            F::Set(name, g) => {
                if matches!(
                    **g,
                    (F::Literal(_)
                        | F::Dot
                        | F::Spread
                        | F::Path(_)
                        | F::Array(_)
                        | F::Object(_)
                        | F::Get(_))
                ) {
                    write!(f, "{g} as ${name}")
                } else {
                    write!(f, "({g}) as ${name}")
                }
            }
            F::Get(name) => write!(f, "${name}"),
            F::Pipe(fs) if fs.is_empty() => Ok(()),
            F::Pipe(fs) => {
                fn pipeseg(p: &F, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    match p {
                        F::Pipe(_) => write!(f, "({p})"),
                        _ => write!(f, "{p}"),
                    }
                }
                pipeseg(&fs[0], f)?;
                for g in &fs[1..] {
                    write!(f, " | ")?;
                    pipeseg(g, f)?;
                }
                Ok(())
            }
            F::Call(name, args) if args.is_empty() => {
                write!(f, "{name}")
            }
            F::Call(name, args) => {
                write!(f, "{name}({}", args[0])?;
                for arg in args {
                    write!(f, ", {arg}")?;
                }
                write!(f, ")")
            }
        }
    }
}

type FStream = Box<dyn Iterator<Item = Context>>;
#[derive(Clone)]
enum FIt<Vs: Iterator> {
    Zero,
    One(Vs::Item),
    Many(Vs),
}
type FIter = FIt<FStream>;
impl<Vs: Iterator> Iterator for FIt<Vs> {
    type Item = <Vs as Iterator>::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let ret: Option<Self::Item>;
        (ret, *self) = match std::mem::replace(self, FIt::Zero) {
            FIt::Zero => (None, FIt::Zero),
            FIt::One(ctx) => (Some(ctx), FIt::Zero),
            FIt::Many(mut fstm) => match fstm.next() {
                Some(ctx) => (Some(ctx), FIt::Many(fstm)),
                None => (None, FIt::Zero),
            },
        };
        ret
    }
}

impl<Vs: Iterator + 'static> FIt<Vs> {
    fn flatter_map<W: 'static>(
        self,
        mut f: impl FnMut(Vs::Item) -> FIt<Box<dyn Iterator<Item = W>>> + 'static,
    ) -> FIt<Box<dyn Iterator<Item = W>>> {
        match self {
            FIt::Zero => FIt::Zero,
            FIt::One(v) => f(v),
            FIt::Many(vs) => FIt::Many(Box::new(vs.flat_map(f)) as Box<dyn Iterator<Item = W>>),
        }
    }
}

impl std::ops::BitOr<F> for Value {
    type Output = FIter;

    fn bitor(self, rhs: F) -> Self::Output {
        rhs.eval((Env::new(), self))
    }
}

mod parse {
    use crate::FPath;

    use super::{FIt, Value as V, F};
    use nom::bytes::complete::{escaped_transform, tag, take, take_while};
    use nom::character::complete::{digit1, multispace0, multispace1};
    use nom::character::is_hex_digit;
    use nom::combinator::{eof, map, map_res, opt, recognize, value};
    use nom::error::ParseError;
    use nom::multi::{count, fold_many0, separated_list0};
    use nom::sequence::{preceded, separated_pair, tuple};
    use nom::{
        branch::alt, character::complete::char, multi::fold_many1, multi::many1,
        multi::separated_list1, sequence::delimited,
    };
    use nom::{Err, IResult, Needed};
    use std::borrow::Cow;
    use std::io::Read;

    fn vnull(rdr: &str) -> IResult<&str, V> {
        let (rdr, _) = tag("null")(rdr)?;
        Ok((rdr, V::Null))
    }
    fn vbool(rdr: &str) -> IResult<&str, V> {
        let (rdr, b) = alt((value(true, tag("true")), value(false, tag("false"))))(rdr)?;
        Ok((rdr, V::Bool(b)))
    }
    fn vnumber(rdr: &str) -> IResult<&str, V> {
        let num = recognize(preceded(opt(char('-')), digit1));
        let num = map_res(num, |ds: &str| ds.parse::<i64>());
        map(num, V::Number)(rdr)
    }
    fn vstring(rdr: &str) -> IResult<&str, V> {
        let (rdr, s) = string(rdr)?;
        Ok((rdr, V::String(s)))
    }
    fn varray(rdr: &str) -> IResult<&str, V> {
        let elems = separated_list0(punct(','), vany);
        map(delimited(punct('['), elems, punct(']')), V::Array)(rdr)
    }
    fn vobject(rdr: &str) -> IResult<&str, V> {
        let kv_pair = separated_pair(string, punct(':'), vany);
        let kv_pairs = separated_list0(char(','), kv_pair);
        let obj = delimited(punct('{'), kv_pairs, punct('}'));
        map(obj, V::Object)(rdr)
    }
    fn vany(rdr: &str) -> IResult<&str, V> {
        alt((vnull, vbool, vnumber, vstring, varray, vobject))(rdr)
    }

    pub(crate) fn json(rdr: &str) -> Result<V, nom::Err<nom::error::Error<&str>>> {
        let (rest, val) = delimited(multispace0, vany, multispace0)(rdr)?;
        eof(rest)?;
        Ok(val)
    }

    // json abnf https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/JSON
    fn string(rdr: &str) -> IResult<&str, String> {
        fn unescaped(c: u64) -> bool {
            (c >= 0x20 && c <= 0x21) || (c >= 0x23 && c <= 0x5b) || (c >= 0x5d && c <= 0x10ffff)
        };
        fn escaped(rdr: &str) -> IResult<&str, char> {
            fn uniesc(rdr: &str) -> IResult<&str, char> {
                let (rdr, _) = tag("\\u")(rdr)?;
                let (rdr, ds) =
                    map_res(take(4usize), |out: &str| u32::from_str_radix(out, 16))(rdr)?;
                let c = char::from_u32(ds);
                if let Some(c) = c {
                    return Ok((rdr, c));
                }
                Result::Err(Err::Error(nom::error::Error::from_error_kind(
                    rdr,
                    nom::error::ErrorKind::Char,
                )))
            }
            alt((
                value('\"', tag("\\\"")),
                value('\\', tag("\\\\")),
                value('/', tag("\\/")),
                value('\x62', tag("\\b")),
                value('\x66', tag("\\f")),
                value('\n', tag("\\n")),
                value('\r', tag("\\r")),
                value('\t', tag("\\t")),
                uniesc,
            ))(rdr)
        }
        let chars = escaped_transform(take_while(|c| unescaped(u64::from(c))), '\\', escaped);
        delimited(char('"'), chars, char('"'))(rdr)
    }
    fn punct<'s>(c: char) -> impl FnMut(&'s str) -> IResult<&str, char> {
        delimited(multispace0, char(c), multispace0)
    }
    fn fident(rdr: &str) -> IResult<&str, &str> {
        take_while(|c: char| c.is_alphanumeric() || c == '_')(rdr)
    }

    fn fliteral(rdr: &str) -> IResult<&str, F> {
        map(alt((vnull, vbool, vnumber, vstring)), F::Literal)(rdr)
    }
    fn fpath(rdr: &str) -> IResult<&str, F> {
        // [_:_]
        let range = separated_pair(opt(fany_left), punct(':'), opt(fany_left));
        let range = map(range, |(a, b)| FPath::Range(a, b));
        // [_]
        let idx = map(fany, FPath::Index);

        let bracket = map(
            delimited(char('['), opt(alt((range, idx))), char(']')),
            |idx| idx.unwrap_or(FPath::Spread),
        );
        let swallow = map(opt(char('?')), |v| v.is_some());
        let path = tuple((bracket, swallow));
        map(preceded(char('.'), many1(path)), F::Path)(rdr)
    }
    fn fdot(rdr: &str) -> IResult<&str, F> {
        let (rdr, _) = char('.')(rdr)?;
        Ok((rdr, F::Dot))
    }
    fn fspread(rdr: &str) -> IResult<&str, F> {
        let (rdr, _) = tag(".[]")(rdr)?;
        Ok((rdr, F::Spread))
    }
    fn fparen(rdr: &str) -> IResult<&str, F> {
        delimited(char('('), fany, char(')'))(rdr)
    }
    fn fget(rdr: &str) -> IResult<&str, F> {
        map(preceded(char('$'), fident), |s| F::Get(s.to_string()))(rdr)
    }
    fn farray(rdr: &str) -> IResult<&str, F> {
        let elems = separated_list0(punct(','), fany);
        map(delimited(punct('['), elems, punct(']')), F::Array)(rdr)
    }
    fn fobject(rdr: &str) -> IResult<&str, F> {
        let kv_pair = separated_pair(string, punct(':'), fany);
        let kv_pairs = separated_list0(char(','), kv_pair);
        let obj = delimited(punct('{'), kv_pairs, punct('}'));
        map(obj, F::Object)(rdr)
    }

    // not left-recursive
    fn fany_left(rdr: &str) -> IResult<&str, F> {
        alt((
            fliteral, fpath, fspread, fdot, fget, farray, fobject, fparen,
        ))(rdr)
    }

    fn fset(rdr: &str) -> IResult<&str, F> {
        let sep = delimited(multispace1, tag("as"), multispace1);
        let set = separated_pair(fany_left, sep, preceded(char('$'), fident));
        map(set, |(f, name)| F::Set(name.to_string(), Box::new(f)))(rdr)
    }
    fn fpipe(rdr: &str) -> IResult<&str, F> {
        let f = alt((fset, fany_left));
        map(separated_list1(punct('|'), f), F::Pipe)(rdr)
    }

    fn fany(rdr: &str) -> IResult<&str, F> {
        fpipe(rdr)
    }

    pub(crate) fn filter(input: &str) -> Result<F, nom::Err<nom::error::Error<&str>>> {
        let (input, ret) = fany(input)?;
        Ok(ret)
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
    (.$([$($f:tt)*])*) => { F::Path(vec![$( (FPath::Index(fjson!($($f)*)), false) ),*]) };
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_macro() {
        let obj = json!({"a": [1,2,{},null], "b": true});
        let f: F = fjson!(
            {"a": [1,2,{},null], "b": true}
            | .["b"] as @foo
            | [ .["a"] | .[] | . ]
            | {"vals": ., "val": .[], "foo": @foo}
        );
        println!("{obj} | {f}");
        for (_, v) in obj | f {
            println!("{v}");
        }
    }
}

fn main() {
    let arg1 = std::env::args().skip(1).next().unwrap_or(String::new());
    let f = match parse::filter(&arg1) {
        Ok(mut f) => {
            f.normalize();
            f
        }
        Err(err) => {
            eprintln!("parse err: {err}");
            std::process::exit(1);
        }
    };

    println!("filter: {f}");

    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .expect("read stdin");
    let val = parse::json(&input).expect("json input");
    for (_, v) in val | f {
        println!("{v}");
    }
}
