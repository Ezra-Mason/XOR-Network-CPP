std::vector<double> SetUpWeights(std::vector<double> weights);
void OutputWeights(std::vector<double> weights);
double Sigmoid(double input);
double Sum(std::vector<double> input);
double GlobalError(std::vector<double> weights, int truth_table[8][4]);
void LogResults(std::vector<double> weights, int truth_table[8][4]);