// MCMC for basic pricing 

// GOAL: compute E[...] to price a derivative
// APPROACH: use law of large numbers to compute E[....] 
//           by taking many draws from a R.V.


// modeled off the example in Mark Joshi's book

#include <iostream>
#include <random>
#include <cmath> 


// input: all pricing variables and a definite value for our random variable 
// output: the price of the option
double OptionPayoff(
    double Expiry,
    double Strike,
    double Spot,
    double Vol,
    double r,
    double randomValue)
{
    double logPrice = (r - 0.5*Vol*Vol)*Expiry + randomValue*sqrt(Expiry)*Vol; 
    double payoff = exp(logPrice)*Spot - Strike;
    return payoff > 0 ? payoff : 0;
}

// more efficient to precompute the part of OptionExpectation 
// that doesnt depend on the random variable
// written this way for pedagogical ease
double OptionExpectation(
    double Expiry,
    double Strike, 
    double Spot,
    double Vol,
    double r,
    unsigned long numSamples
)
{
    // get a Gaussian distribution 
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, 1.0); // mean 0, stddev 1

    double sum = 0.0;
    for (unsigned long i = 0; i < numSamples; ++i) {
        double randomValue = dist(gen);  
        sum += OptionPayoff(Expiry, Strike, Spot, Vol, r, randomValue);
    }
    return sum / numSamples;
}

int main() {
    double Expiry = 1.0;
    double Strike = 100.0;
    double Spot = 100.0;
    double Vol = 0.2;
    double r = 0.05;
    unsigned long numSamples = 100000;

    double price = OptionExpectation(Expiry, Strike, Spot, Vol, r, numSamples);
    std::cout << "Option Price: " << price << std::endl;

    return 0;
}
