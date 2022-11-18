


pub fn csv_loader(filename: &str) -> Result<Vec<Vec<String>>, std::io::Error> {

    const ABSOLUTE_PATH: &str = "/Users/andremarques/Desktop/";

    let path = format!("{}{}", ABSOLUTE_PATH, filename);

    let mut file = std::fs::File::open(&path)?;

    use std::io::Read;
    let mut data = String::new();
    file.read_to_string(&mut data)?;

    let data = data.split_terminator("\r\n").into_iter().map(|row| {
        row.split(',').map(|string| string.to_owned()).collect::<Vec<_>>()
    }).collect::<Vec<_>>();

    Ok(data)
}