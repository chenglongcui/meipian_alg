if __name__ == '__main__':
    file = open("../data/result_ad_message.csv", "w", encoding='gbk')
    with open("../data/知识付费站内信文案.csv", "r") as f:
        for line in f.readlines():
            # line = line.strip("\n")
            print(line)
            file.writelines(line)
    file.close()
