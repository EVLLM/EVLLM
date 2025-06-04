ÇALIÞTIRMA REHBERÝ FROM T T T T T TUNCERRRRRRRRRRRRR

eval "$(/truba/sw/centos7.9/lib/anaconda3/2023.03/bin/conda shell.bash hook)"

Step by step:

Data prep:
	1. output_<sayý> yazan ve içinde datalar bulunan dosyayý EZSR klasörüne al
	2. klasöre aldýðýn output_<sayý> klasörünü  h5_to_npz.py file'ýndan input olarak seç ve outputu da o dosyada output_<sayý>_npz isminde bir folder'a kaydettir.( bu kýsým bir scriptle  otomatize edilebilir ya da direkt argüman alacak þekilde ayarlanabilir, ya da manuel devam edebilir. Sana býrakýyorum o kýsmý kafana göre takýl)
	3. Her þey hazýrsa konsola "sinfo" yazýp akya ve barbundaki "idle" serverlara bakýlýr.
	4. Idle server olan kümenin ayarlarý "convert_h5.sh" dosyasýna girilir. ( folder isimleri argüman olarak verilecekse .sh dosyasýna da eklenmeli)
	5. terminalden "sbatch convert_h5.sh" yazarak çalýþtýrýlýr. Herhangi bir hatada Tuncere yazýlýr.
	6. Kod çýktý olarak bütün .h5 eventleri .npz eventlere çevirip modele vermeye hazýr hale getirecek.
EZSR inference:
	1. input olarak output_<sayý>_npz folder'ý ve output'u kaydedeceði event_features_<sayý> folderi "ezsr.py" dosyasýna yazýlýr(ayný þekilde argüman olarak ya da script ile otomatize edilebilir sana kalmýþ)
	2. "ezsr.py" koduna göz atýlýr, pretrained model path'i, dosya folderlerý doðru mu kontrol edilir.
	3. input folder'i hazýrsa yine sinfo'dan akya ve barbun kontrol edilir ve uygun olana göre "ezsr.sh" dosyasý ayarlanýr
	4. terminale "sbatch ezsr.sh" yazarak çalýþtýrýlýr
	5. Hata alýnýrsa Tuncer'e yazýlýr
	6. Inference time 1000 .npz baþýna yaklaþýk 30-40 dk sürüyor iþlem sonunda iþlemin kaç dk sürdüðü ve iþlenen dosyalar vs yazýyor
