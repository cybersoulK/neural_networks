
pub struct Console;

impl Console {

    pub fn new() -> Self {
        Self
    }
    
    pub fn run<L>(&self, mut console_loop: L)
    where 
        L: FnMut(String, Vec<String>) + 'static 
    {
        println!("welcome to Console");

        loop {
            let mut inputs = String::new();
            std::io::stdin().read_line(&mut inputs).unwrap();
            
            let mut inputs = inputs.split_whitespace().into_iter().map(|str| str.to_owned()).collect::<Vec<_>>();
            
            let command = if !inputs.is_empty() {
                inputs.remove(0)
            }
            else { String::new() };

            let params = inputs;


            match command.as_str() {
                "help" => {
                    println!("commands: exit, clear");
                }
                "exit" => {
                    break;
                }
                "clear" => {
                    clear_console();
                }

                _ => console_loop(command, params),
            }
        }
    }   
}


pub fn clear_console() {
    println!("\x1B[2J\x1B[1;1H");
}