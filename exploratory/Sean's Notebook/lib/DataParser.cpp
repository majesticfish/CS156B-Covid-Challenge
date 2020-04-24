#include "DataParser.h"
#include "DateParser.h"
#include "DateParser.cpp"

using namespace std;

string HOMEDIR = "/home/sean/Dropbox/MyDocuments/Programming/CS156B/CS156B-Covid-Challenge"

class CumDeathGenerator : TargetGenerator {
public: 
    void init() {
        ifstream file("HOMEDIR" + "/data/us/covid/deaths.csv");

        string line;
        getline(file, line);
        vector<int> dates;        
    }
    double getY(string fips, int day) {
        return 0;
    }
};
