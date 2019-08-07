

for i in range(5):
    print('名字を入力してください')
    family_name = input()
    print('名前を入力してください')
    given_name = input()
    name = family_name + given_name
    print('あなたの名前は', name)

    if len(name) >= 10:
        print('You have long name!!')

