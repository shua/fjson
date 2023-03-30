#![allow(unused)]

use std::{fmt::Display, io::Read, rc::Rc};

use codespan_reporting::{
    diagnostic::{Diagnostic, Label},
    files::SimpleFile,
    term::{
        self,
        termcolor::{ColorChoice, StandardStream},
    },
};
use lalrpop_util::lalrpop_mod;

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

    fn add(&self, rhs: &Value) -> Result<Value, String> {
        match (self, rhs) {
            (Value::Null, v) | (v, Value::Null) => Ok(v.clone()),
            (Value::Number(n), Value::Number(m)) => Ok(Value::Number(n + m)),
            (Value::String(s), Value::String(t)) => Ok(Value::String({
                let mut s = s.clone();
                s.push_str(t);
                s
            })),
            (Value::Array(a), Value::Array(b)) => Ok(Value::Array({
                let mut a = a.clone();
                a.extend(b.iter().cloned());
                a
            })),
            (Value::Object(a), Value::Object(b)) => Ok(Value::Object({
                let mut a = a.clone();
                for (k, v) in b {
                    if a.iter().any(|(k2, _)| k == k2) {
                        continue;
                    }
                    a.push((k.clone(), v.clone()));
                }
                a
            })),
            (a, b) => Err(format!(
                "error: {} ({a}) and {} ({b}) cannot be added",
                a.kind(),
                b.kind()
            )),
        }
    }

    fn mul(&self, rhs: &Value) -> Result<Value, String> {
        match (self, rhs) {
            (Value::Number(n), Value::String(s)) | (Value::String(s), Value::Number(n)) => {
                if *n >= 1 {
                    Ok(Value::String(s.repeat(usize::try_from(*n).unwrap())))
                } else {
                    Ok(Value::Null)
                }
            }
            (Value::Number(n), Value::Number(m)) => Ok(Value::Number(n * m)),
            (a, b) => Err(format!(
                "error: {} ({a}) and {} ({b}) cannot be multiplied",
                a.kind(),
                b.kind(),
            )),
        }
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
                let idx = usize::try_from(nm[0]).unwrap()..usize::try_from(nm[1]).unwrap();
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
impl Clone for Env {
    fn clone(&self) -> Self {
        match self {
            Env::Empty => Env::Empty,
            Env::Borrowed(p) => Env::Borrowed(p.clone()),
            Env::Owned { ext, int } => Env::Borrowed(Rc::new(Env::Owned {
                ext: ext.clone(),
                int: int.clone(),
            })),
        }
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
            Env::Owned { int, .. } => int.push(kv),
        }
    }
    fn get(&self, key: impl AsRef<str>) -> Option<Value> {
        match self {
            Env::Empty => None,
            Env::Borrowed(ext) => ext.get(key),
            Env::Owned { ext, int } => int
                .iter()
                .find(|(k, _)| k.as_str() == key.as_ref())
                .map(|(_, v)| v.clone())
                .or_else(|| ext.get(key)),
        }
    }

    fn borrowed(self) -> Env {
        match self {
            Env::Owned { .. } => Env::Borrowed(Rc::new(self)),
            env => env,
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
        match self {
            FPath::Index(f) => f.normalize(),
            FPath::Range(s, e) => {
                s.as_mut().map(F::normalize);
                e.as_mut().map(F::normalize);
            }
            FPath::Spread => {}
            FPath::Field(_) => {}
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum FOp {
    Pipe,
    Add,
    Mul,
}

impl Display for FOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FOp::Pipe => write!(f, "|"),
            FOp::Add => write!(f, "+"),
            FOp::Mul => write!(f, "*"),
        }
    }
}

#[derive(Debug, Clone)]
enum F {
    Literal(Value),
    Array(Vec<F>),
    Object(Vec<(String, F)>),
    Path(Vec<(FPath, bool)>),
    Set(String, Box<F>),
    Get(String),
    Binop(FOp, Vec<F>),
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
            FPath::Index(F::Path(ps)) if ps.is_empty() => cloned_with_env(val.get(val0), env),
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
            FPath::Spread => match val {
                Value::Array(mut es) => match es.len() {
                    0 => FIter::Zero,
                    1 => FIter::One((env, es.pop().unwrap())),
                    _ => FIter::Many(Box::new(es.into_iter().map(move |v| (env.clone(), v)))),
                },
                Value::Object(mut fs) => match fs.len() {
                    0 => FIter::Zero,
                    1 => FIter::One((env, fs.pop().unwrap().1)),
                    _ => FIter::Many(Box::new(fs.into_iter().map(move |(_, v)| (env.clone(), v)))),
                },
                _ => errexit!("cannot iterate over {} ({})", val.kind(), val),
            },
            FPath::Field(name) => cloned_with_env(val.get(name.as_str()), env),
        }
    }

    fn precedence(&self) -> usize {
        match self {
            F::Literal(_) => usize::MAX,
            F::Array(_) => usize::MAX,
            F::Object(_) => usize::MAX,
            F::Path(_) => usize::MAX,
            F::Set(_, _) => 2,
            F::Get(_) => usize::MAX,
            F::Binop(FOp::Pipe, _) => 1,
            F::Binop(FOp::Add, _) => 3,
            F::Binop(FOp::Mul, _) => 4,
            F::Call(_, _) => usize::MAX,
        }
    }

    fn normalize(&mut self) {
        match self {
            F::Literal(_) | F::Get(_) => {}
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
                for (_, f) in kvs.iter_mut() {
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
            F::Path(ps) => ps.iter_mut().map(|(p, _)| p.normalize()).collect(),
            F::Set(_, f) => f.normalize(),
            F::Binop(_, ref mut fs) if fs.len() == 1 => {
                let mut f = fs.pop().unwrap();
                f.normalize();
                *self = f;
            }
            F::Binop(_, fs) => fs.iter_mut().map(F::normalize).collect(),
            F::Call(_, args) => args.iter_mut().map(F::normalize).collect(),
        }
    }

    fn eval(&self, ctx: Context) -> FIter {
        match self {
            // "foo", 1, null, true
            F::Literal(val) => FIter::One((ctx.0, val.clone())),
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
                        fit => FIt::Many(Box::new(fit.map(move |(_, v)| vec![(hd.0.clone(), v)]))),
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
                        (kvs, _) => FIt::Many(Box::new(kvs.flat_map(move |kvs| {
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
            // .
            F::Path(fs) if fs.is_empty() => FIter::One(ctx),
            // .[f1][f2] ..
            F::Path(fs) => {
                let mut it = fs.iter();
                let first: FIter = F::fpath(ctx.clone(), &ctx, it.next().unwrap());
                it.fold(first, |it, f| {
                    let ctx0 = ctx.clone();
                    let f = f.clone();
                    it.flatter_map(move |ctx| F::fpath(ctx, &ctx0, &f))
                })
            }
            // f as $name
            F::Set(name, fval) => {
                let it = fval.eval(ctx.clone());
                let name = name.clone();
                it.flatter_map(move |(_, v)| {
                    let mut env = ctx.0.clone();
                    env.set(&name, v);
                    FIter::One((env, ctx.1.clone()))
                })
            }
            // $name
            F::Get(name) => {
                let val = ctx.0.get(name).unwrap_or(Value::Null);
                FIter::One((ctx.0, val))
            }
            // ? weird case
            F::Binop(_, fs) if fs.is_empty() => FIter::Zero,
            F::Binop(_, fs) if fs.len() == 1 => fs[0].eval(ctx),
            // f1 | f2 | ..
            F::Binop(FOp::Pipe, fs) => {
                let mut it = fs.iter();
                let fstm: FIter = it.next().unwrap().eval(ctx);
                it.fold(fstm, |it, f| {
                    let f = f.clone();
                    it.flatter_map(move |ctx| f.eval(ctx))
                })
            }
            // f1 _ f2 _ ...
            F::Binop(op @ (FOp::Add | FOp::Mul), fs) => {
                let mut it = fs.iter().cloned();
                let hd = it.next().unwrap();
                let mut hd = hd.eval(ctx.clone());
                let binop = if op == &FOp::Add {
                    Value::add
                } else {
                    Value::mul
                };
                for f in it {
                    let ctx = ctx.clone();
                    hd = hd.flatter_map(move |(_, v)| {
                        let env0 = ctx.0.clone();
                        f.eval(ctx.clone())
                            .flatter_map(move |(_, w)| match binop(&v, &w) {
                                Ok(v) => FIter::One((env0.clone(), v)),
                                Err(err) => {
                                    eprintln!("{err}");
                                    std::process::exit(1);
                                }
                            })
                    })
                }
                hd
            }

            // function defined in wider scope
            F::Call(_name, _args) => {
                todo!()
            }
        }
    }
}

struct FPathDisplay<'p>(&'p [(FPath, bool)]);
impl<'p> Display for FPathDisplay<'p> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.is_empty() {
            return write!(f, ".");
        }

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
                if g.precedence() > self.precedence() {
                    write!(f, "{g} as ${name}")
                } else {
                    write!(f, "({g}) as ${name}")
                }
            }
            F::Get(name) => write!(f, "${name}"),
            F::Binop(_, fs) if fs.is_empty() => Ok(()),
            F::Binop(op, fs) => {
                fn factor(g: &F, p: usize, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    if g.precedence() > p {
                        write!(f, "{g}")
                    } else {
                        write!(f, "({g})")
                    }
                }
                let p = self.precedence();
                factor(&fs[0], p, f)?;
                for g in &fs[1..] {
                    write!(f, " {op} ")?;
                    factor(g, p, f)?;
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

lalrpop_mod!(grammar);

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

    (.) => { F::Path(vec![]) };
    (.[]) => { F::Path(vec![(FPath::Spread, false)]) };
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
    ($f:tt | $($g:tt)|*) => { F::Binop(FOp::Pipe, vec![ fjson!($f), $(fjson!($g)),*]) };
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
    (@@pipe [$($f:expr),*]) => { F::Binop(FOp::Pipe, vec![ $($f),* ]) };
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

fn show_parse_error(
    name: &str,
    input: &str,
    err: &lalrpop_util::ParseError<usize, grammar::Token, &str>,
) {
    use std::io::Write as _;
    let writer = StandardStream::stderr(ColorChoice::Auto);
    let config = codespan_reporting::term::Config::default();
    let file = SimpleFile::new(name, input);

    let diag: Diagnostic<()>;
    fn exp_str(exp: &[String]) -> String {
        match exp.len() {
            0 => format!("expected eof"),
            1 => format!("expected {}", exp[0]),
            _ => {
                let mut out = String::from("expected one of: ");
                out.push_str(&exp[0]);
                for s in &exp[1..exp.len() - 1] {
                    out.push_str(", ");
                    out.push_str(s);
                }
                out.push_str(", or ");
                out.push_str(&exp[exp.len() - 1]);
                out
            }
        }
    }
    match err {
        lalrpop_util::ParseError::InvalidToken { location } => {
            diag = Diagnostic::error()
                .with_message("parse error: invalid token")
                .with_labels(vec![Label::primary((), *location..*location)]);
        }
        lalrpop_util::ParseError::UnrecognizedEOF { location, expected } => {
            diag = Diagnostic::error()
                .with_message("parse error: unrecognized eof")
                .with_labels(vec![Label::primary((), *location..*location)])
                .with_notes(vec![exp_str(expected)]);
        }
        lalrpop_util::ParseError::UnrecognizedToken { token, expected } => {
            diag = Diagnostic::error()
                .with_message("parse error: unrecognized token")
                .with_labels(vec![Label::primary((), token.0..token.2)])
                .with_notes(vec![exp_str(expected)]);
        }
        lalrpop_util::ParseError::ExtraToken { token } => {
            diag = Diagnostic::error()
                .with_message("parse error: unexpected token")
                .with_labels(vec![Label::primary((), token.0..token.2)])
                .with_notes(vec![exp_str(&[])]);
        }
        lalrpop_util::ParseError::User { error } => {
            diag = Diagnostic::error().with_message(format!("parse error: {error}"));
        }
    }

    term::emit(&mut writer.lock(), &config, &file, &diag).unwrap();
}

fn main() {
    let arg1 = std::env::args().skip(1).next().unwrap_or(String::new());
    let f = match grammar::FilterParser::new().parse(&arg1) {
        Ok(mut f) => {
            f.normalize();
            f
        }
        Err(err) => {
            show_parse_error("<arg[1]>", &arg1, &err);
            std::process::exit(1);
        }
    };

    println!("filter: {f}");

    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .expect("read stdin");
    let val = match grammar::ValueParser::new().parse(&input) {
        Ok(v) => v,
        Err(err) => {
            show_parse_error("<stdin>", &input, &err);
            std::process::exit(1);
        }
    };
    for (_, v) in val | f {
        println!("{v}");
    }
}
