import numpy as np
import collections
from sklearn.naive_bayes import GaussianNB

#fungsi split input dan target digunakan untuk menggunakan libary naive bayes yakni dengan memisahkan x1, x2 dengan kelas
def memisahkan_target_dan_input(data):
    input = data.T[:2].T
    kelas = data.T[-1].T
    return input, kelas

#melakukan grain model dengan library python
def klasifikasi_naivebayes(bootstrap):
    #variable input dan target diperuntukkan untuk menampung fungsi dari split input dan data yang berfungsi untuk dipanggil oleh naive bayes library
    input, target = memisahkan_target_dan_input(bootstrap)
    #model = data train yang diolah oleh library naive bayes
    model = GaussianNB().fit(input, target)
    return model

#melakukan random pada dataset
def score_random(dataset):
    return np.random.randint(dataset)

#mengenerate bootstrap yakni merupakan salah satu model dari bagging
def make_Boostrap(dataset):
    #menginisiasi variable bootstrap dengan numpy zeros
    bootstrap = np.zeros(dataset.shape)
    #melakukan perulangan dari i hingga jumlah dari dataset
    for i in range(dataset.shape[0]):
        #melakukan nilai random pada score
        iterasi = score_random(dataset.shape[0])
        bootstrap[i] = dataset[iterasi]
    return bootstrap

# mengetahui nilai yang sering muncul
def voting(nilai_akhir):
    hasil = []
    for i, output in enumerate(nilai_akhir.T):
        temp = collections.Counter(output)
        most = temp.most_common(1)[0][0]
        hasil.append(most)
    print(hasil)
    return hasil

def validasi(input_validasi, models):
    outputs = []
    for i in models:
        simpan = i.predict(input_validasi)
        outputs.append(simpan)
    outputs = np.array(outputs)
    return outputs

#membuka file csv dari data train
def load_datatrain():
    #menghapus bagian header dengan sintak skiprows = 1
    main_dataset = np.loadtxt('TrainsetTugas4ML.csv', skiprows=1, delimiter=',')
    return main_dataset, main_dataset

#membuka file csv dari data test
def load_datatest():
    data_test = np.genfromtxt('TestsetTugas4ML.csv', delimiter=',')[1:-1]
    return data_test

#membuat model bootstrap setelah melakukan klasifikasi pada naive bayes
def membuat_model(dataset):
    models = []
    for i in range(bootstraps.shape[0]):
        bootstraps[i] = make_Boostrap(dataset)
        models.append(klasifikasi_naivebayes(bootstraps[i]))
    print(models)
    return models

# melakukan prediksi terhadap data train
def predict(input):
    fix = []
    for i in models:
        a = i.predict(input)
        fix.append(a)
    outputs = np.array(fix)
    print(outputs)
    return outputs

if __name__ == '__main__':
    #meload data train lalu ditampung ke dalam dataset dan data_validasi
    dataset, data_validasi = load_datatrain()
    print(dataset)
    print(data_validasi)
    #memampung numpy zeros yang berisi jumlah dari banyaknya barisxkolomxdimensiarray
    bootstraps = np.zeros((100, dataset.shape[0], dataset.shape[1]))
    # membuat boostrap dan membuat model yang akan di train
    models = []
    models = membuat_model(dataset)
    # melakukan validasi terhadap data test dengan data train
    input_validasi, target_validasi = memisahkan_target_dan_input(data_validasi)
    nilai_akhir = validasi(input_validasi, models)
    # menampung hasil nilai dari fungsi yang sering muncul
    hasil = voting(nilai_akhir)
    # membuka csv dari datatest
    data_test = load_datatest()
    # melakukan prediksi terhadap data train
    input, target = memisahkan_target_dan_input(data_test)
    #menampung hasil prediksi yang diberikan data input yakni merupakan data test yang di split
    nilai_akhir = predict(input)
    # mencari nilai yang sama dan dihitung lalu dimasukkan ke dalam data test yakni kelas tersebut yang masih kosong
    for i, j in enumerate(nilai_akhir.T):
        simpan = collections.Counter(j)
        data_test[i][2] = simpan.most_common(1)[0][0]
        print(data_test[i][2])
    #menyimpan hasil data test
    np.savetxt('TebakanTugas4ML.csv', data_test, delimiter=',')