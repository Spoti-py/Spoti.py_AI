for i in range(3):
    TOTAL = []
    REPO = []
    count = 1
    n = input()
    for i in range(8):
        TOTAL.append(n[i])
    for i in range(1, 8):
        if TOTAL[i-1] == TOTAL[i]:
            count += 1
            REPO.append(count)
        else:
            count = 1
            REPO.append(count)
    print(max(REPO))