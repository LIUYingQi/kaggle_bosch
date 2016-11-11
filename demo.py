import csv
file_path = '/home/liuyingqi/Desktop/kaggle_project'
count = 0

# load general info from Data_strcture.csv file
with open(file_path+'/BOSCH/test_categorical.csv','rb') as general_info:
    reader = csv.reader(general_info)
    for row in reader:
        count+=1
print count
#     print reader.next()
#     print reader.next()
#     print reader.next()
#     print reader.next()
#     print reader.next()
#     print reader.next()
#     print reader.next()
#     print reader.next()
# #
# with open(file_path+'/BOSCH/test_numeric.csv','rb') as general_info:
#     # reader = csv.reader(general_info)
#     # print reader.next()
#     # print reader.next()
#     # print reader.next()
#     # print reader.next()
#     # print reader.next()
#     # print reader.next()
#     # print reader.next()
    # print reader.next()
# #
# # with open(file_path+'/BOSCH/train_date.csv','rb') as general_info:
# #     reader = csv.reader(general_info)
# #     print reader.next()
# #     print reader.next()
# #     print reader.next()
# #     print reader.next()
#
#     print len(reader.next())
