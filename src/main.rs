use bullet::{data::Data, arch::NNUEParams, gd_tune};

pub const NET_NAME: &str = "maiden";

fn main() -> std::io::Result<()> {
    let file_name = String::from("wha.epd");

    // initialise data
    let mut data = Data::default();
    data.1 = 6;
    data.add_contents(&file_name);

    // provide starting parameters
    let mut params = NNUEParams::default();


    // carry out tuning
    gd_tune(&data, &mut params, 100, 0.05, NET_NAME);

    params.write_to_bin(&format!("{NET_NAME}-final.bin"))?;

    // exit
    Ok(())
}