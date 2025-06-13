#include <payoff.h>
#include <cmath>
#include <algorithm>


// constructor definition
PayOff::PayOff(double Strike_, OptionType type_)
    : Strike(Strike_), type(type_) {}

double PayOff::operator()(double Spot) const {
    switch (type) {
        case Call:
            return std::max(Spot - Strike, 0.0); // Call option payoff
        case Put:
            return std::max(Strike - Spot, 0.0); // Put option payoff
        default:
            throw std::invalid_argument("Unknown option type");
    }

}