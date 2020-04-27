#include <bits/stdc++.h>

using namespace std;

class TargetGenerator {
public:
    virtual void init();
    virtual double getY(string fips, int day);
};

class DeltaDeathsGenerator : TargetGenerator {
public:
    void init();
    double getY(string fips, int day);
};

class CumDeathsGenerator : TargetGenerator {
public:
    void init();
    double getY(string fips, int day);
private:
    map<string, vector<int>> data;
};

class DeltaCasesGenerator : TargetGenerator {
public:
    void init();
    double getY(string fips, int day);
};

class CumCasesGenerator : TargetGenerator {
public:
    void init();
    double getY(string fips, int day);
private:
    map<string, vector<int>> data;
};

void printXY(vector<string> features, TargetGenerator Y);
