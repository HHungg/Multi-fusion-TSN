import pickle
import numpy as np

def random_position(length, num_seq, rgb=True, data_type=1):
    length = length - 1
    divide = length / num_seq
    train_render = []

    if length > 3*10*data_type:
        for i in range(num_seq):
            if i < num_seq - 1:
                if(i==0):
                    k = np.random.randint(1,divide*(i+1)-9*data_type+1)
                else:
                    k = np.random.randint(divide*i+1,divide*(i+1)-9*data_type+1)
            else:
                k = np.random.randint(divide*i+1,length-9*data_type+1)
            train_render.append(k)
    elif (length > 10*data_type):
        divide = (length-10) / num_seq
        for i in range(num_seq):
            if i < num_seq - 1:
                if(i==0):
                    k = np.random.randint(1,divide*(i+1)+1)
                else:
                    k = np.random.randint(divide*i+1,divide*(i+1)+1)
            else:
                k = np.random.randint(divide*i+1,length-9*data_type+1)
            train_render.append(k)
            train_render.sort()
    else:
        train_render = np.ones((num_seq,), dtype=int)
    return train_render

# open a file, where you stored the pickled data
file = open('C:/Users/HungLM/Desktop/2/results/incept229_twostream1_1_lstm256_34e_cr1.pickle', 'rb', encoding='utf-8')

# dump information to that file
data = pickle.load(file, fix_imports=True)

# close the file
file.close()

# print(data)

# print('Showing the pickled data:')

cnt = 0
for i in range(9):
    new_data = []
    for item in data:
        print('The data ', cnt, ' is : ', item)
        cnt += 1
        # if cnt > 700:
        #     break
    #     length = item[4]
    #     render = random_position(length, num_seq=3)
    #     item[1] = render
    #     print(item)
    #     new_data.append(item)

    # out_file = r'C:/Users/HungLM/OneDrive/Documents/20172/CNN/TSN-LSTM/database/ucf101-test3-split1-test0'+ str(i) +'.pickle'
    # with open(out_file,'wb') as f2:
    #     pickle.dump(new_data,f2, protocol=2)
    


