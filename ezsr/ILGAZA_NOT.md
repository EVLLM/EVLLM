�ALI�TIRMA REHBER� FROM T T T T T TUNCERRRRRRRRRRRRR

eval "$(/truba/sw/centos7.9/lib/anaconda3/2023.03/bin/conda shell.bash hook)"

Step by step:

Data prep:
	1. output_<say�> yazan ve i�inde datalar bulunan dosyay� EZSR klas�r�ne al
	2. klas�re ald���n output_<say�> klas�r�n�  h5_to_npz.py file'�ndan input olarak se� ve outputu da o dosyada output_<say�>_npz isminde bir folder'a kaydettir.( bu k�s�m bir scriptle  otomatize edilebilir ya da direkt arg�man alacak �ekilde ayarlanabilir, ya da manuel devam edebilir. Sana b�rak�yorum o k�sm� kafana g�re tak�l)
	3. Her �ey haz�rsa konsola "sinfo" yaz�p akya ve barbundaki "idle" serverlara bak�l�r.
	4. Idle server olan k�menin ayarlar� "convert_h5.sh" dosyas�na girilir. ( folder isimleri arg�man olarak verilecekse .sh dosyas�na da eklenmeli)
	5. terminalden "sbatch convert_h5.sh" yazarak �al��t�r�l�r. Herhangi bir hatada Tuncere yaz�l�r.
	6. Kod ��kt� olarak b�t�n .h5 eventleri .npz eventlere �evirip modele vermeye haz�r hale getirecek.
EZSR inference:
	1. input olarak output_<say�>_npz folder'� ve output'u kaydedece�i event_features_<say�> folderi "ezsr.py" dosyas�na yaz�l�r(ayn� �ekilde arg�man olarak ya da script ile otomatize edilebilir sana kalm��)
	2. "ezsr.py" koduna g�z at�l�r, pretrained model path'i, dosya folderler� do�ru mu kontrol edilir.
	3. input folder'i haz�rsa yine sinfo'dan akya ve barbun kontrol edilir ve uygun olana g�re "ezsr.sh" dosyas� ayarlan�r
	4. terminale "sbatch ezsr.sh" yazarak �al��t�r�l�r
	5. Hata al�n�rsa Tuncer'e yaz�l�r
	6. Inference time 1000 .npz ba��na yakla��k 30-40 dk s�r�yor i�lem sonunda i�lemin ka� dk s�rd��� ve i�lenen dosyalar vs yaz�yor
