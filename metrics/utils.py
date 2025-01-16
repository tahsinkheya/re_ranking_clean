def map_age(age):
    if age < 18:
        return 0
    elif 18 <= age <= 24:
        return 1
    elif 25 <= age <= 34:
        return 2
    elif 35 <= age <= 44:
        return 3
    elif 45 <= age <= 49:
        return 4
    elif 50 <= age <= 55:
        return 5
    else:
        return 6
