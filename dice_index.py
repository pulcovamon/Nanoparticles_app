'''File for computing Dice and Jacard index'''

true_positive = 4 + 4 + 6 + 30 + 33 + 22
false_positive = 1 + 0 + 1 + 4 + 12 + 4
false_negative = 0 + 0 + 6 + 6
dice_index = [0, 0, 0, 0, 0]
jacard_index = [0, 0, 0, 0, 0]

dice_index[0] = 2* true_positive / (2 * true_positive + false_positive + false_negative)

jacard_index[0] = true_positive / (true_positive + false_positive + false_negative)

true_positive = 3 + 2 + 7
false_positive = 0 + 0 + 0
false_negative = 0 + 0 + 0

dice_index[1] = 2* true_positive / (2 * true_positive + false_positive + false_negative)

jacard_index[1] = true_positive / (true_positive + false_positive + false_negative)

true_positive = 12 + 10 + 6 + 4 + 4
false_positive = 0
false_negative = 3 + 1

dice_index[2] = 2* true_positive / (2 * true_positive + false_positive + false_negative)

jacard_index[2] = true_positive / (true_positive + false_positive + false_negative)

true_positive = 8 + 19 + 5 + 31 + 19
false_positive = 3
false_negative = 52

dice_index[3] = 2* true_positive / (2 * true_positive + false_positive + false_negative)

jacard_index[3] = true_positive / (true_positive + false_positive + false_negative)

true_positive = 8 + 2 + 6 + 7 + 4 + 8 + 11 + 7 + 7 + 11 + 2 + 2 + 7 + 12 + 4 + 10 + 5 + 12 + 2 + 2 + 10 + 5 + 5 + 14 + 13 + 14 + 8 + 6 + 12 + 3 + 1 + 11
false_positive = 1 + 2 + 1 + 2 + 1 + 1 + 3 + 2 + 1 + 1 + 2 + 1 + 5 + 1 + 4 + 3 + 1 + 3 + 1
false_negative = 1 + 1 + 1 + 1 + 1 + 2 + 7 + 8 + 2 + 4 + 1 + 1 + 8 + 10 + 9 + 6 + 4 + 6 + 4 + 3

dice_index[4] = 2* true_positive / (2 * true_positive + false_positive + false_negative)

jacard_index[4] = true_positive / (true_positive + false_positive + false_negative)

dice_index_sum = sum(dice_index) / 5
jacard_index_sum = sum(jacard_index) / 5
print(dice_index_sum, dice_index)
print(jacard_index_sum, jacard_index)