extern crate rand_distr;
extern crate rand;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

struct PricingData {
    expiry: f32,
    strike: f32,
    spot: f32,
    volatility: f32,
    r: f32
}

fn price_option(data: &PricingData, random_value: f32) -> f32 {
    let fixed_part_log = (data.r - 0.5 * data.volatility.powi(2)) * data.expiry;
    let random_part_log = random_value * data.volatility * data.expiry.sqrt();
    let payoff = data.spot * (fixed_part_log + random_part_log).exp() - data.strike;
    if payoff > 0.0 {
        payoff
    } else {
        0.0
    }
}


fn mc_pricing_simple(data: &PricingData, n: u32) -> f32 {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = thread_rng();
    let mut sum = 0.0;

    for _ in 0..n {
        let random_value = normal.sample(&mut rng);
        sum += price_option(&data, random_value);
    }

    (sum / n as f32) * (-data.r * data.expiry).exp()
}


fn main() {
    let data = PricingData {
        expiry: 1.0,
        strike: 100.0,
        spot: 100.0,
        volatility: 0.2,
        r: 0.05
    };

    let n = 1000000; // Number of simulations
    let price = mc_pricing_simple(&data, n);
    println!("The estimated option price is: {:.2}", price);
}