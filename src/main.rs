fn usage() {
    println!(
        r#"usage: jp [-chlr] PTR
parse json input from stdin, and extract values specified by PTR

  PTR  json pointer (/foo/bar/0) https://www.rfc-editor.org/rfc/rfc6901
  -c   compact output
  -h   print usage string
  -l   listy formatted output
  -r   raw strings

{} {} by {}"#,
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        env!("CARGO_PKG_AUTHORS")
    )
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut compact = false;
    let mut raw = false;
    let mut list = false;
    let mut ptr = "".to_string();
    while let Some(arg) = args.next() {
        if arg.len() >= 2 && &arg[0..1] == "-" {
            for f in arg.chars().skip(1) {
                match f {
                    'c' => compact = true,
                    'h' => {
                        usage();
                        return;
                    }
                    'l' => list = true,
                    'r' => raw = true,
                    f => {
                        eprintln!("error: unrecognized option -{f}");
                        std::process::exit(1);
                    }
                }
            }
        } else {
            ptr = arg;
        }
    }
    let json: serde_json::Value =
        serde_json::from_reader(std::io::stdin()).expect("json from stdin");
    let json = match json.pointer(ptr.as_str()) {
        Some(v) => v,
        None => &serde_json::Value::Null,
    };

    match json {
        serde_json::Value::Null if raw => println!(),
        serde_json::Value::Null => println!("null"),
        serde_json::Value::Bool(b) => println!("{b}"),
        serde_json::Value::Number(n) => println!("{n}"),
        serde_json::Value::String(s) if raw => println!("{s}"),
        serde_json::Value::String(s) => println!("{s:?}"),
        json if compact => {
            serde_json::to_writer(std::io::stdout(), &json).unwrap();
            println!();
        }
        json if list => {
            print_listy(json);
        }
        json => {
            serde_json::to_writer_pretty(std::io::stdout(), &json).unwrap();
            println!();
        }
    }
}

fn print_listy(json: &serde_json::Value) {
    fn inner(json: &serde_json::Value, indent: usize) {
        match json {
            serde_json::Value::Null => println!("null"),
            serde_json::Value::Bool(b) => println!("{b}"),
            serde_json::Value::Number(n) => println!("{n}"),
            serde_json::Value::String(s) => println!("{s:?}"),
            serde_json::Value::Array(a) => inner_arr(&a, indent),
            serde_json::Value::Object(o) => inner_map(o, indent),
        }
    }
    fn inner_arr(arr: &[serde_json::Value], indent: usize) {
        print!("[");
        for (i, v) in arr.iter().enumerate() {
            if i == 0 {
                print!(" ");
            } else {
                print!(", ");
            }
            inner(v, indent + 1);
            for _ in 0..indent {
                print!("  ");
            }
        }
        println!("]");
    }
    fn inner_map(objs: &serde_json::Map<String, serde_json::Value>, indent: usize) {
        print!("{{");
        for (i, (k, v)) in objs.iter().enumerate() {
            if i == 0 {
                print!(" ");
            } else {
                print!(", ");
            }
            print!("{k:?}: ");
            if v.is_array() || v.is_object() {
                println!();
                for _ in 0..(indent + 1) {
                    print!("  ");
                }
            }
            inner(v, indent + 1);
            for _ in 0..indent {
                print!("  ");
            }
        }
        println!("}}");
    }
    inner(json, 0)
}
