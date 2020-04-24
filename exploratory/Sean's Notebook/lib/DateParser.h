#include <iostream>
 
using namespace std;
 
struct Date 
{ 
 int d, m, y; 
};

int countLeapYears(Date d);

int getDifference(Date dt1, Date dt2);

int date_to_int(string day);
