import numpy as np
import pickle as pkl

def predict(x):
    def my_convolve2d(a, conv_filter):
        submatrices = np.array([
            [a[:-2, :-2], a[:-2, 1:-1], a[:-2, 2:]],
            [a[1:-1, :-2], a[1:-1, 1:-1], a[1:-1, 2:]],
            [a[2:, :-2], a[2:, 1:-1], a[2:, 2:]]])
        multiplied_subs = np.einsum('ij,ijkl->ijkl', conv_filter, submatrices)
        return np.sum(np.sum(multiplied_subs, axis=-3), axis=-3)
    def hog(image):
        nwin_x = 7
        nwin_y = 7
        B = 11
        (L, C) = np.shape(image)
        H = np.zeros(shape=(nwin_x * nwin_y * B, 1))
        m = np.sqrt(L / 2.0)
        if C is 1:
            raise NotImplementedError
        step_x = np.floor(C / (nwin_x + 1))
        step_y = np.floor(L / (nwin_y + 1))
        cont = 0
        hx = np.array([[1, 0, -1]])
        hy = np.array([[-1], [0], [1]])
        grad_xr = my_convolve2d(image, hx)
        grad_yu = my_convolve2d(image, hy)
        angles = np.arctan2(grad_yu, grad_xr)
        magnit = np.sqrt((grad_yu ** 2 + grad_xr ** 2))
        for n in range(nwin_y):
            for m in range(nwin_x):
                cont += 1
                angles2 = angles[int(n * step_y):int((n + 2) * step_y), int(m * step_x):int((m + 2) * step_x)]
                magnit2 = magnit[int(n * step_y):int((n + 2) * step_y), int(m * step_x):int((m + 2) * step_x)]
                v_angles = angles2.ravel()
                v_magnit = magnit2.ravel()
                K = np.shape(v_angles)[0]
                bin = 0
                H2 = np.zeros(shape=(B, 1))
                for ang_lim in np.arange(start=-np.pi + 2 * np.pi / B, stop=np.pi + 2 * np.pi / B, step=2 * np.pi / B):
                    check = v_angles < ang_lim
                    v_angles = (v_angles * (~check)) + (check) * 100
                    H2[bin] += np.sum(v_magnit * check)
                    bin += 1
                H2 = H2 / (np.linalg.norm(H2) + 0.01)
                H[(cont - 1) * B:cont * B] = H2
        return H
    def macierzTeta(x, w):
        licznik = sigmoid(x @ w.transpose())
        return licznik
    def sigmoid(x):
        x_e = x - np.max(x, axis=1)[:, None]
        licznik = np.exp(x_e)
        mianownik = np.sum(licznik, axis=1)[:, None]
        return licznik / mianownik
    def hog_x(x):
        dlugosc = len(x[:, 0])
        x_new = np.zeros([dlugosc, 539])
        for i in range(0, dlugosc):
            x_new[i] = hog(np.reshape(x[i], (56, 56))).reshape(-1)
        return x_new


    w = pkl.load(open('macierzNowaW.pkl', mode='rb'))
    x_test = hog_x(x)
    duzaMacierzTeta = macierzTeta(x_test, w)
    Y_wynik = np.argmax(duzaMacierzTeta, axis=1)
    Y_wynik = np.reshape(Y_wynik, (x.shape[0], 1))
    return Y_wynik








