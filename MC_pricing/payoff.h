#ifndef PAYOFF_H

#define PAYOFF_H

class PayOff {
    // methods accessible from outside the class 
    public: 
        enum OptionType { Call, Put }; // option types, call and puts supported to start
        PayOff(double Strike_, OptionType type_);
        double operator()(double Spot) const;

    private:
        double Strike;
        OptionType type;
};
#endif 
