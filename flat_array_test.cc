#include "flat_array.h"

#include <cassert>
#include <cstdint>
#include <iostream>

using namespace std;

void test_at();
void test_field();

int main() {
    test_at();
    test_field();

    cout << "+----------------+" << endl;
    cout << "| ALL TESTS PASS |" << endl;
    cout << "+----------------+" << endl;
}


//         +---+---+---+---+    +---+---+---+---+    +---+---+---+---+
//       4 |116|117|118|119|  4 |216|217|218|219|  4 |316|317|318|319|
//         +---+---+---+---+    +---+---+---+---+    +---+---+---+---+
//       3 |112|113|114|115|  3 |212|213|214|215|  3 |312|313|314|315|
//         +---+---+---+---+    +---+---+---+---+    +---+---+---+---+
//       2 |108|109|110|111|  2 |208|209|210|211|  2 |308|309|310|311|
//         +---+---+---+---+    +---+---+---+---+    +---+---+---+---+
//       1 |104|105|106|107|  1 |204|205|206|207|  1 |304|305|306|307|
//         +---+---+---+---+    +---+---+---+---+    +---+---+---+---+
//       0 |100|101|102|103|  0 |200|201|202|203|  0 |300|301|302|303|
//         +---+---+---+---+    +---+---+---+---+    +---+---+---+---+
//           0   1   2   3        0   1   2   3        0   1   2   3
//
//  Returns a 5x4 array with three fields with values shown above.
int *sample_array(int *nx, int *ny, int *num_fields) {
    assert (nx && ny && num_fields);

    *nx = 4;
    *ny = 5;
    *num_fields = 3;
    int *xs = (int *)malloc(*nx * *ny * *num_fields * sizeof(int));
    int *xs0 = &xs[0 * *nx * *ny];
    int *xs1 = &xs[1 * *nx * *ny];
    int *xs2 = &xs[2 * *nx * *ny];

    for (int i = 0; i < *nx * *ny; ++i) {
        xs0[i] = 100 + i;
        xs1[i] = 200 + i;
        xs2[i] = 300 + i;
    }
    return xs;
}

void test_at() {
    int nx, ny, num_fields;
    int *xs = sample_array(&nx, &ny, &num_fields);

    assert (*flat_array::at(xs, nx, ny, 0, 0) == 100);
    assert (*flat_array::at(xs, nx, ny, 1, 0) == 101);
    assert (*flat_array::at(xs, nx, ny, 2, 0) == 102);
    assert (*flat_array::at(xs, nx, ny, 3, 0) == 103);
    assert (*flat_array::at(xs, nx, ny, 0, 1) == 104);
    assert (*flat_array::at(xs, nx, ny, 1, 1) == 105);
    assert (*flat_array::at(xs, nx, ny, 2, 1) == 106);
    assert (*flat_array::at(xs, nx, ny, 3, 1) == 107);
    assert (*flat_array::at(xs, nx, ny, 0, 2) == 108);
    assert (*flat_array::at(xs, nx, ny, 1, 2) == 109);
    assert (*flat_array::at(xs, nx, ny, 2, 2) == 110);
    assert (*flat_array::at(xs, nx, ny, 3, 2) == 111);
    assert (*flat_array::at(xs, nx, ny, 0, 3) == 112);
    assert (*flat_array::at(xs, nx, ny, 1, 3) == 113);
    assert (*flat_array::at(xs, nx, ny, 2, 3) == 114);
    assert (*flat_array::at(xs, nx, ny, 3, 3) == 115);
    assert (*flat_array::at(xs, nx, ny, 0, 4) == 116);
    assert (*flat_array::at(xs, nx, ny, 1, 4) == 117);
    assert (*flat_array::at(xs, nx, ny, 2, 4) == 118);
    assert (*flat_array::at(xs, nx, ny, 3, 4) == 119);

    free(xs);
}

void test_field() {
    int nx, ny, num_fields;
    int *xs = sample_array(&nx, &ny, &num_fields);
    int *xs0 = flat_array::field(xs, nx, ny, 0);
    int *xs1 = flat_array::field(xs, nx, ny, 1);
    int *xs2 = flat_array::field(xs, nx, ny, 2);

    // field 1
    assert (*flat_array::at(xs0, nx, ny, 0, 0) == 100);
    assert (*flat_array::at(xs0, nx, ny, 1, 0) == 101);
    assert (*flat_array::at(xs0, nx, ny, 2, 0) == 102);
    assert (*flat_array::at(xs0, nx, ny, 3, 0) == 103);
    assert (*flat_array::at(xs0, nx, ny, 0, 1) == 104);
    assert (*flat_array::at(xs0, nx, ny, 1, 1) == 105);
    assert (*flat_array::at(xs0, nx, ny, 2, 1) == 106);
    assert (*flat_array::at(xs0, nx, ny, 3, 1) == 107);
    assert (*flat_array::at(xs0, nx, ny, 0, 2) == 108);
    assert (*flat_array::at(xs0, nx, ny, 1, 2) == 109);
    assert (*flat_array::at(xs0, nx, ny, 2, 2) == 110);
    assert (*flat_array::at(xs0, nx, ny, 3, 2) == 111);
    assert (*flat_array::at(xs0, nx, ny, 0, 3) == 112);
    assert (*flat_array::at(xs0, nx, ny, 1, 3) == 113);
    assert (*flat_array::at(xs0, nx, ny, 2, 3) == 114);
    assert (*flat_array::at(xs0, nx, ny, 3, 3) == 115);
    assert (*flat_array::at(xs0, nx, ny, 0, 4) == 116);
    assert (*flat_array::at(xs0, nx, ny, 1, 4) == 117);
    assert (*flat_array::at(xs0, nx, ny, 2, 4) == 118);
    assert (*flat_array::at(xs0, nx, ny, 3, 4) == 119);

    // field 2
    assert (*flat_array::at(xs1, nx, ny, 0, 0) == 200);
    assert (*flat_array::at(xs1, nx, ny, 1, 0) == 201);
    assert (*flat_array::at(xs1, nx, ny, 2, 0) == 202);
    assert (*flat_array::at(xs1, nx, ny, 3, 0) == 203);
    assert (*flat_array::at(xs1, nx, ny, 0, 1) == 204);
    assert (*flat_array::at(xs1, nx, ny, 1, 1) == 205);
    assert (*flat_array::at(xs1, nx, ny, 2, 1) == 206);
    assert (*flat_array::at(xs1, nx, ny, 3, 1) == 207);
    assert (*flat_array::at(xs1, nx, ny, 0, 2) == 208);
    assert (*flat_array::at(xs1, nx, ny, 1, 2) == 209);
    assert (*flat_array::at(xs1, nx, ny, 2, 2) == 210);
    assert (*flat_array::at(xs1, nx, ny, 3, 2) == 211);
    assert (*flat_array::at(xs1, nx, ny, 0, 3) == 212);
    assert (*flat_array::at(xs1, nx, ny, 1, 3) == 213);
    assert (*flat_array::at(xs1, nx, ny, 2, 3) == 214);
    assert (*flat_array::at(xs1, nx, ny, 3, 3) == 215);
    assert (*flat_array::at(xs1, nx, ny, 0, 4) == 216);
    assert (*flat_array::at(xs1, nx, ny, 1, 4) == 217);
    assert (*flat_array::at(xs1, nx, ny, 2, 4) == 218);
    assert (*flat_array::at(xs1, nx, ny, 3, 4) == 219);

    // field 3
    assert (*flat_array::at(xs2, nx, ny, 0, 0) == 300);
    assert (*flat_array::at(xs2, nx, ny, 1, 0) == 301);
    assert (*flat_array::at(xs2, nx, ny, 2, 0) == 302);
    assert (*flat_array::at(xs2, nx, ny, 3, 0) == 303);
    assert (*flat_array::at(xs2, nx, ny, 0, 1) == 304);
    assert (*flat_array::at(xs2, nx, ny, 1, 1) == 305);
    assert (*flat_array::at(xs2, nx, ny, 2, 1) == 306);
    assert (*flat_array::at(xs2, nx, ny, 3, 1) == 307);
    assert (*flat_array::at(xs2, nx, ny, 0, 2) == 308);
    assert (*flat_array::at(xs2, nx, ny, 1, 2) == 309);
    assert (*flat_array::at(xs2, nx, ny, 2, 2) == 310);
    assert (*flat_array::at(xs2, nx, ny, 3, 2) == 311);
    assert (*flat_array::at(xs2, nx, ny, 0, 3) == 312);
    assert (*flat_array::at(xs2, nx, ny, 1, 3) == 313);
    assert (*flat_array::at(xs2, nx, ny, 2, 3) == 314);
    assert (*flat_array::at(xs2, nx, ny, 3, 3) == 315);
    assert (*flat_array::at(xs2, nx, ny, 0, 4) == 316);
    assert (*flat_array::at(xs2, nx, ny, 1, 4) == 317);
    assert (*flat_array::at(xs2, nx, ny, 2, 4) == 318);
    assert (*flat_array::at(xs2, nx, ny, 3, 4) == 319);

    free(xs);
}
