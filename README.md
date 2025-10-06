Reprise_code is a 2023 code which is not working (the aoa calcultion is wrong)
Test_1 is a working version of this code, very fast but not the most accurate
Test_2 is not working, bad implementation of the aoa calculation
Test_3 is not working, tries to do both a 1ms shoot (fireed from the gound) and a 300ms shot (shoot from a plane), the result does not work
Test_4 is slower with 2 passes but more accurate (also optimise the angle of launch) , only aoa applied after the apogee (works)
Test_5 is the same as test_4 but takes the wing lift and drag intoo acount (before only the body was taken into account)
Test_6 is also trying to opmise the aoa on the ascending phase, it's a lot slower but more accurate
Test_7 introduces a stall model (aoa >20°) that does not exist in the previous code
Test_8
