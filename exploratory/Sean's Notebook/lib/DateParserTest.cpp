#include "DateParser.h"
#include <iostream>
#include <assert.h>

using namespace std;

int main() {
    cout << "Testing DateParser.cpp\n";
    assert(date_to_int("2020-01-01") == 0);
    assert(date_to_int("2020-01-21") == 20);
    assert(date_to_int("2020-02-01") == 31);
    assert(date_to_int("2020-03-01") == 60);
    assert(date_to_int("2020-04-01") == 91);

    cout << "All tests pass!\n";
}
