from flask import Flask, render_template_string, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.class_weight import compute_class_weight
import random, os

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 1. LOAD & TRAIN RANDOM FOREST
# ============================================================
def train_classification_model():
    df = pd.read_csv(os.path.join(BASE, "Untitled spreadsheet - Data_Label_Gizi_ZScore.csv"))

    df.rename(columns={
        'Konsumsi daging setiap hari': 'daging',
        'Aktivitas fisik (olahraga) dalam satu hari': 'durasi_olahraga',
        'Bagaimana dengan makanan cepat saji yang anda konsumsi?': 'fast_food',
        'Berapa jam durasi tidur malam Anda setiap harinya?': 'tidur',
        'Berapa lama waktu yang Anda habiskan untuk menatap layar (HP, TV, Laptop) sambil duduk atau rebahan dalam sehari?': 'screen_time',
        'Bagaimana pola konsumsi sayur dan buah Anda dalam porsi makan sehari-hari?': 'sayur_buah',
        'Dalam satu hari, berapa porsi/gelas minuman manis (seperti es teh manis, kopi susu gula aren, boba, minuman kemasan, soda) yang Anda minum?': 'minuman_manis',
        'Seberapa rutin Anda mengonsumsi susu atau produk olahannya (seperti yoghurt manis, susu full cream, keju) dalam sehari?': 'susu'
    }, inplace=True)

    df = df.drop_duplicates().dropna()

    # Normalize categorical values
    replace_map = {
        'Tidak Pernah': 'tidak pernah', 'Sesekali': 'kadang', 'Sering': 'sering',
        'Kadang-Kadang': 'kadang', 'Kadang kadang': 'kadang',
        'Kadang - Kadang': 'kadang', 'Kadang -Kadang': 'kadang'
    }
    for col in ['daging', 'fast_food', 'sayur_buah', 'minuman_manis', 'susu']:
        df[col] = df[col].replace(replace_map)

    df['durasi_olahraga'] = df['durasi_olahraga'].replace({
        '15 - 30 menit': '15-30 menit', '15  - 30 menit': '15-30 menit'
    })

    # FIX 1: Normalize tidur values (dataset has both '<6 jam' and '< 6 jam')
    df['tidur'] = df['tidur'].str.replace(' ', '').str.replace('-', ' - ')
    df['tidur'] = df['tidur'].replace({
        '<6jam': '< 6 jam', '6-8jam': '6 - 8 jam', '>8jam': '> 8 jam',
        '6-8 jam': '6 - 8 jam',
    })

    # FIX 2: Normalize screen_time values
    df['screen_time'] = df['screen_time'].str.strip()
    df['screen_time'] = df['screen_time'].replace({
        '> 3 jam': '> 3 jam', '< 1 jam': '< 1 jam',
        '1 -3 jam': '1 - 3 jam', '1 - 3 jam': '1 - 3 jam'
    })

    fitur_num = ['Usia', 'Tinggi_Badan_cm', 'Berat_Badan_kg']
    fitur_cat = ['Jenis_Kelamin', 'daging', 'durasi_olahraga', 'fast_food',
                 'tidur', 'screen_time', 'sayur_buah', 'minuman_manis', 'susu']
    X = df[fitur_num + fitur_cat]
    y = df['Label_Gizi']

    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    class_weight_dict = dict(zip(classes, weights))

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), fitur_num),
        ('cat', OneHotEncoder(handle_unknown='ignore'), fitur_cat)
    ])

    pipeline = Pipeline([
        ('prep', preprocessor),
        ('model', RandomForestClassifier(
            n_estimators=200,       # FIX 3: lebih banyak tree → lebih stabil
            max_depth=6,            # sedikit lebih dalam
            min_samples_split=8,
            min_samples_leaf=2,
            random_state=42, n_jobs=-1,
            class_weight=class_weight_dict
        ))
    ])
    pipeline.fit(X, y)
    return pipeline, df['Label_Gizi'].unique().tolist()

model_rf, label_classes = train_classification_model()

# ============================================================
# 2. KNN REKOMENDASI MENU
# ============================================================

# ── WHITELIST: Kata kunci nama makanan yang dianggap FAMILIAR ──────────────
# Makanan utama yang familiar dan umum dikonsumsi sehari-hari di Indonesia
WHITELIST_MAKANAN = [
    # Nasi & karbohidrat pokok
    'nasi', 'beras', 'bihun', 'mie', 'lontong', 'ketupat', 'bubur',
    # Ayam
    'ayam goreng', 'ayam bakar', 'ayam panggang', 'ayam kalasan',
    'ayam ampela', 'ayam usus',
    # Ikan populer
    'ikan mas', 'ikan lele', 'ikan bandeng', 'ikan tongkol', 'ikan nila',
    'ikan mujair', 'ikan patin', 'ikan bakar', 'ikan goreng', 'ikan asin bilis',
    'gulai ikan', 'pindang ikan',
    # Daging sapi & kambing populer
    'daging sapi', 'rendang', 'soto', 'empal goreng masakan', 'bakso',
    'semur daging', 'daging kambing',
    # Tahu & tempe
    'tahu goreng', 'tahu telur', 'tempe goreng', 'tempe kedelai murni',
    # Telur
    'telur ayam', 'telur ayam dadar', 'telur ayam ceplok', 'telur ayam kampung',
    'telur bebek',
    # Sayur
    'bayam tumis', 'gudeg sayur', 'sayur garu', 'singkong goreng',
    'ubi jalar goreng',
    # Seafood familiar
    'cumi-cumi goreng', 'udang goreng', 'kepiting',
]

# Snack yang familiar
WHITELIST_SNACK = [
    # Buah-buahan umum
    'apel', 'pisang ambon', 'pisang goreng', 'pisang lampung', 'pisang mas',
    'pisang raja', 'mangga golek', 'mangga indramayu', 'mangga kopek', 'mangga muda',
    'jambu',
    # Kacang populer
    'kacang tanah rebus', 'kacang hijau rebus', 'kacang kedelai rebus',
    'kacang merah kering rebus', 'kacang bogor rebus',
    # Jajanan familiar
    'pisang goreng', 'risoles', 'roti putih', 'roti gambang',
    'kue apem', 'kue lumpur', 'kue talam', 'kue sus', 'kue bugis',
    'martabak', 'getuk pisang', 'es mambo',
    # Susu
    'susu sapi', 'susu kambing', 'susu kental manis',
    # Lain-lain familiar
    'pempek', 'nasi rames', 'oncom pepes', 'ikan bandeng presto',
    'ikan mas pepes', 'ikan mujair pepes', 'sardines',
]

def filter_familiar(df, whitelist):
    """Kembalikan baris yang namanya mengandung salah satu kata kunci whitelist (case-insensitive)."""
    mask = df['name'].str.lower().apply(
        lambda n: any(kw in n for kw in whitelist)
    )
    filtered = df[mask].reset_index(drop=True)
    # Fallback: jika terlalu sedikit item, kembalikan dataframe asli tanpa filter
    if len(filtered) < 5:
        return df.reset_index(drop=True)
    return filtered

# ── Konfigurasi pembagian menu per waktu makan ──────────────────────────
MEAL_KKAL_RATIO = {
    'sarapan': 0.20,
    'siang':   0.35,
    'malam':   0.30,
}
SNACK_KKAL_RATIO = 0.15
MEAL_LABEL = {
    'sarapan': ('🍳', 'Sarapan'),
    'siang':   ('🍛', 'Makan Siang'),
    'malam':   ('🌙', 'Makan Malam'),
}

N_KANDIDAT_PER_WAKTU_MAKAN = 5000   # jumlah kandidat menu per waktu makan
RASIO_KANDIDAT_MENU_LENGKAP = 0.30  # porsi kandidat "Menu Lengkap" di antara kandidat yang dibuat
N_KANDIDAT_SNACK = 5000

# ── KNN K=5 + evaluasi/voting (meniru jurnal referensi: Nafi & Mujiyono, 2025) ──
K_NEIGHBORS = 5                # K pada KNN: ambil 5 tetangga terdekat, bukan cuma 1
AMBANG_SANGAT_SESUAI = 0.15     # rata-rata deviasi macro <=15% dari target -> "Sangat Sesuai"
AMBANG_CUKUP_SESUAI  = 0.35     # <=35% -> "Cukup Sesuai", di atas itu -> "Kurang Sesuai"


def _split_kategori_makanan(mb_df):
    """
    Pisahkan dataset makanan berat berdasarkan kolom 'kategori', lalu terapkan
    whitelist familiar PER KATEGORI (bukan di seluruh dataset sekaligus).
    Ini penting: kalau whitelist diterapkan ke seluruh dataset dulu, satu
    kategori yang datanya tipis (mis. Sayur) bisa habis/kosong walau total
    baris yang lolos whitelist masih terlihat cukup — akibatnya kombinasi
    Karbo+Lauk+Sayur gagal terbentuk dan sistem selalu jatuh ke Menu Lengkap.
    """
    kat = mb_df['kategori'].astype(str).str.strip().str.lower()
    mentah = {
        'karbo':   mb_df[kat == 'karbo'].reset_index(drop=True),
        'lauk':    mb_df[kat == 'lauk'].reset_index(drop=True),
        'sayur':   mb_df[kat == 'sayur'].reset_index(drop=True),
        'lengkap': mb_df[kat == 'menu lengkap'].reset_index(drop=True),
    }
    return {
        nama: filter_familiar(df, WHITELIST_MAKANAN) if len(df) else df
        for nama, df in mentah.items()
    }


def _hitung_gizi_baris(baris):
    """Jumlahkan kalori & makronutrien dari sekumpulan baris (list of Series)."""
    cal  = sum(b['calories']     for b in baris)
    carb = sum(b['carbohydrate'] for b in baris)
    prot = sum(b['proteins']     for b in baris)
    fat  = sum(b['fat']          for b in baris)
    return cal, carb, prot, fat


def _generate_kandidat_menu(kategori, n_kandidat, rasio_lengkap):
    """
    Bangun kandidat menu untuk satu waktu makan.
    - Menu biasa  : 1 Karbo + 1 Lauk + 1 Sayur (realistis, tidak dobel karbo)
    - Menu lengkap: 1 item saja (mis. Mie Ayam, Nasi Goreng, Soto Ayam) — sudah satu paket
    """
    karbo, lauk, sayur, lengkap = (
        kategori['karbo'], kategori['lauk'], kategori['sayur'], kategori['lengkap']
    )

    n_lengkap = int(n_kandidat * rasio_lengkap) if len(lengkap) > 0 else 0
    n_biasa   = n_kandidat - n_lengkap

    kandidat, gizi = [], []

    if len(karbo) and len(lauk) and len(sayur):
        for _ in range(n_biasa):
            k = karbo.iloc[random.randrange(len(karbo))]
            l = lauk.iloc[random.randrange(len(lauk))]
            s = sayur.iloc[random.randrange(len(sayur))]
            baris = [k, l, s]
            cal, carb, prot, fat = _hitung_gizi_baris(baris)
            kandidat.append({'items': baris, 'is_lengkap': False, 'cal': cal})
            gizi.append([carb, prot, fat])

    for _ in range(n_lengkap):
        m = lengkap.iloc[random.randrange(len(lengkap))]
        cal, carb, prot, fat = _hitung_gizi_baris([m])
        kandidat.append({'items': [m], 'is_lengkap': True, 'cal': cal})
        gizi.append([carb, prot, fat])

    return kandidat, np.array(gizi)


def _kunci_dedup_menu(menu):
    """Kunci dedup yang TIDAK peduli urutan item (mis. 'Bakwan + Susu Kedelai'
    dan 'Susu Kedelai + Bakwan' harus dianggap kombinasi yang SAMA)."""
    if isinstance(menu, dict) and 'items' in menu:
        return frozenset(it['name'] for it in menu['items'])
    return frozenset(menu['name'].tolist())


def _nama_ringkas_menu(menu):
    """Nama singkat gabungan item dalam satu kandidat menu, dipakai untuk dedup & tampilan.
    Mendukung 2 bentuk kandidat: dict {'items':[...]} (menu makan berat) ATAU
    DataFrame 2 baris (kombinasi snack)."""
    if isinstance(menu, dict) and 'items' in menu:
        return " + ".join(it['name'] for it in menu['items'])
    return " + ".join(menu['name'].tolist())


def _evaluasi_kandidat(gizi_kandidat, target_macro):
    """
    Evaluasi KUALITATIF seberapa dekat gizi kandidat (karbo, protein, lemak)
    terhadap target — meniru kolom 'Kesesuaian' pada Tabel 2 jurnal referensi
    (Nafi & Mujiyono, 2025), yang mengevaluasi TIAP tetangga terdekat satu per
    satu, bukan cuma mengandalkan total jarak Euclidean.
    """
    dev = [abs(g - t) / max(t, 1e-6) for g, t in zip(gizi_kandidat, target_macro)]
    rata2 = sum(dev) / len(dev)
    if rata2 <= AMBANG_SANGAT_SESUAI:
        label = 'Sangat Sesuai'
    elif rata2 <= AMBANG_CUKUP_SESUAI:
        label = 'Cukup Sesuai'
    else:
        label = 'Kurang Sesuai'
    return label, rata2


def _knn_top_k(kandidat, gizi, target_macro, subset_idx, k=K_NEIGHBORS):
    """
    PERSIS seperti Tabel 2 jurnal referensi ("Menu Terdekat"): ambil K=5
    tetangga terdekat (bukan cuma 1) berdasarkan jarak Euclidean, lalu
    evaluasi kesesuaian tiap tetangga satu per satu.

    CATATAN: karena kandidat dibangun via random sampling (bisa ada beberapa
    baris dengan kombinasi menu yang PERSIS SAMA), diambil dulu tetangga
    mentah dalam jumlah lebih banyak lalu di-dedup berdasarkan nama menu,
    supaya top-K yang dikembalikan benar-benar K MENU YANG BERBEDA — bukan
    menu yang sama terulang beberapa kali.
    """
    if subset_idx is None or len(subset_idx) == 0:
        return []
    k_query = min(max(k * 20, k), len(subset_idx))
    knn = NearestNeighbors(n_neighbors=k_query, metric='euclidean')
    knn.fit(gizi[subset_idx])
    jarak, idx = knn.kneighbors(np.array([target_macro]))

    hasil, nama_terlihat = [], set()
    for d, i in zip(jarak[0], idx[0]):
        real_idx = subset_idx[i]
        kunci = _kunci_dedup_menu(kandidat[real_idx])
        if kunci in nama_terlihat:
            continue
        nama_terlihat.add(kunci)
        label, deviasi = _evaluasi_kandidat(gizi[real_idx], target_macro)
        hasil.append({
            'rank': len(hasil) + 1,
            'real_idx': real_idx,
            'menu': kandidat[real_idx],
            'gizi': gizi[real_idx],
            'jarak': float(d),
            'kesesuaian': label,
            'deviasi': deviasi,
        })
        if len(hasil) >= k:
            break
    return hasil


def _pilih_pemenang_voting(top_k):
    """
    Tahap 'voting' K=5 (meniru Bagian 3.3 jurnal referensi: 'Hasil Rekomendasi
    Berdasarkan Voting K=5'): dari 5 tetangga terdekat, menangkan kandidat
    dengan jarak TERKECIL di antara yang berlabel BUKAN 'Kurang Sesuai'.
    Kalau kelima-limanya 'Kurang Sesuai' (kasus langka bila dataset kategori
    itu sangat terbatas), tetap ambil yang jaraknya paling kecil (fallback).
    """
    if not top_k:
        return None
    kandidat_baik = [t for t in top_k if t['kesesuaian'] != 'Kurang Sesuai']
    kumpulan = kandidat_baik if kandidat_baik else top_k
    return min(kumpulan, key=lambda t: t['jarak'])


def _knn_kandidat_terbaik(kandidat, gizi, target_macro, subset_idx, k=K_NEIGHBORS):
    """
    Cari kandidat terbaik dari K=5 tetangga terdekat (dibatasi subset_idx)
    memakai KNN Euclidean + tahap evaluasi/voting, PERSIS seperti metode di
    jurnal referensi (Nafi & Mujiyono, 2025) — bukan lagi cuma ambil 1
    tetangga terdekat. Mengembalikan pemenang (menu, gizi, jarak) DAN daftar
    lengkap K tetangganya (dipakai untuk ditampilkan di UI sebagai bukti
    proses KNN K=5-nya).
    """
    top_k = _knn_top_k(kandidat, gizi, target_macro, subset_idx, k)
    if not top_k:
        return None, None, None, []
    pemenang = _pilih_pemenang_voting(top_k)
    return pemenang['menu'], pemenang['gizi'], pemenang['jarak'], top_k


def _rekomendasi_satu_waktu_makan(kategori, target_kalori_meal):
    """
    Untuk 1 waktu makan: buat ribuan kandidat, hitung gizi tiap kandidat,
    lalu cari yang paling dekat dengan target (Euclidean/KNN) — dipisah
    antara opsi "menu biasa" dan opsi "menu lengkap" agar bisa dibandingkan.

    Selain hasil terbaik, kandidat mentah (kandidat/gizi/idx) ikut disimpan
    di return value karena dibutuhkan nanti oleh proses anti-duplikat
    (mencari kandidat terbaik BERIKUTNYA jika pilihan pertama ternyata sudah
    dipakai di waktu makan lain).
    """
    target_carb = (target_kalori_meal * 0.55) / 4
    target_prot = (target_kalori_meal * 0.15) / 4
    target_fat  = (target_kalori_meal * 0.30) / 9
    target_macro = [target_carb, target_prot, target_fat]

    kandidat, gizi = _generate_kandidat_menu(
        kategori, N_KANDIDAT_PER_WAKTU_MAKAN, RASIO_KANDIDAT_MENU_LENGKAP
    )

    idx_biasa   = np.array([i for i, k in enumerate(kandidat) if not k['is_lengkap']])
    idx_lengkap = np.array([i for i, k in enumerate(kandidat) if k['is_lengkap']])

    m_biasa,   g_biasa,   e_biasa,   top_k_biasa   = _knn_kandidat_terbaik(kandidat, gizi, target_macro, idx_biasa)
    m_lengkap, g_lengkap, e_lengkap, top_k_lengkap = _knn_kandidat_terbaik(kandidat, gizi, target_macro, idx_lengkap)

    return {
        'target_cal': target_kalori_meal,
        'target_macro': target_macro,
        'kandidat': kandidat,
        'gizi': gizi,
        'idx_biasa': idx_biasa,
        'idx_lengkap': idx_lengkap,
        'biasa':   {'menu': m_biasa,   'gizi': g_biasa,   'error': e_biasa,   'top_k': top_k_biasa}   if m_biasa   is not None else None,
        'lengkap': {'menu': m_lengkap, 'gizi': g_lengkap, 'error': e_lengkap, 'top_k': top_k_lengkap} if m_lengkap is not None else None,
    }


def _tentukan_waktu_menu_lengkap(hasil_per_waktu):
    """
    Aturan: Menu Lengkap boleh muncul MAKSIMAL 1x sehari.
    Tentukan 1 waktu makan (jika ada) di mana Menu Lengkap paling menguntungkan
    (error KNN-nya lebih kecil dibanding opsi biasa) — waktu makan lain tetap
    memakai kombinasi Karbo + Lauk + Sayur.
    """
    kandidat_tukar = []
    for waktu, hasil in hasil_per_waktu.items():
        if hasil['lengkap'] and hasil['biasa']:
            improvement = hasil['biasa']['error'] - hasil['lengkap']['error']
            if improvement > 0:
                kandidat_tukar.append((improvement, waktu))

    if kandidat_tukar:
        kandidat_tukar.sort(reverse=True)          # improvement terbesar menang
        return kandidat_tukar[0][1]
    return None


def _nama_item_menu(menu_terpilih):
    """
    Kumpulan NAMA makanan di dalam sebuah kandidat menu (tanpa memedulikan
    urutan). Dipakai untuk mendeteksi apakah suatu ITEM (bukan cuma kombinasi
    persisnya) sudah pernah dipakai di waktu makan lain.
    """
    return set(item['name'] for item in menu_terpilih['items'])


def _cari_alternatif_tanpa_duplikat(info_waktu, tipe, item_terpakai):
    """
    Cari kandidat terbaik (tipe: 'biasa' atau 'lengkap') untuk satu waktu makan,
    TAPI hanya dari kandidat yang SEMUA item-nya belum pernah dipakai di waktu
    makan sebelumnya (`item_terpakai`). Artinya: begitu satu makanan (mis.
    "Nasi Putih") sudah keluar di sarapan, makanan itu tidak akan dipilih lagi
    di siang/malam — walau dipasangkan dengan lauk/sayur yang berbeda.
    Dipakai saat pilihan pertama (hasil KNN awal) ternyata memakai item yang
    sudah dipakai di waktu makan sebelumnya.
    Return None jika tidak ada kandidat tersisa yang seluruh item-nya unik
    (kandidat terlalu sedikit).
    """
    subset_idx = info_waktu['idx_biasa'] if tipe == 'biasa' else info_waktu['idx_lengkap']
    if subset_idx is None or len(subset_idx) == 0:
        return None

    kandidat, gizi, target_macro = info_waktu['kandidat'], info_waktu['gizi'], info_waktu['target_macro']
    idx_unik = np.array([
        i for i in subset_idx
        if not (_nama_item_menu(kandidat[i]) & item_terpakai)
    ])
    if len(idx_unik) == 0:
        return None

    menu, gizi_terpilih, error, top_k = _knn_kandidat_terbaik(kandidat, gizi, target_macro, idx_unik)
    return {'menu': menu, 'gizi': gizi_terpilih, 'error': error, 'top_k': top_k}


def _susun_menu_final_tanpa_duplikat(hasil_per_waktu, waktu_menu_lengkap):
    """
    Susun menu final untuk Sarapan → Siang → Malam SECARA BERURUTAN, sambil
    memastikan tidak ada MAKANAN (item) yang sama dipakai lebih dari satu kali
    dalam sehari. Contoh: kalau sarapan sudah memakai Nasi Putih + Telur Dadar
    + Tumis Kangkung, maka ketiga makanan itu (baik sendiri-sendiri maupun
    dalam kombinasi lain) tidak akan muncul lagi di menu siang atau malam.

    Jika kandidat terbaik untuk suatu waktu makan ternyata memakai item yang
    sudah dipakai di waktu makan sebelumnya, sistem mencari kandidat terbaik
    BERIKUTNYA (via KNN, dibatasi ke kandidat yang seluruh item-nya belum
    pernah dipakai) untuk menggantikannya.
    """
    item_terpakai = set()
    menu_final = {}

    for waktu in ['sarapan', 'siang', 'malam']:
        info = hasil_per_waktu[waktu]
        tipe = 'lengkap' if waktu == waktu_menu_lengkap else 'biasa'
        pilihan = info[tipe] or info['biasa'] or info['lengkap']  # fallback jika kategori kosong

        nama_item = _nama_item_menu(pilihan['menu'])
        if nama_item & item_terpakai:
            alternatif = _cari_alternatif_tanpa_duplikat(info, tipe, item_terpakai)
            if alternatif is not None:
                pilihan = alternatif
                nama_item = _nama_item_menu(pilihan['menu'])
            # Jika benar-benar tidak ada kandidat unik tersisa (kasus langka,
            # dataset kategori sangat sedikit), item duplikat tetap
            # dipertahankan sebagai upaya terakhir agar sistem tidak gagal.

        item_terpakai |= nama_item
        menu_final[waktu] = pilihan

    return menu_final


def _pilih_snack_knn(snack_df, target_macro, n_kandidat=N_KANDIDAT_SNACK, k=K_NEIGHBORS):
    """Pilih 2 snack (KNN K=5 + evaluasi/voting Euclidean) agar mendekati target kalori/makro camilan (15% dari total kalori harian)."""
    kombinasi, gizi = [], []
    for _ in range(n_kandidat):
        idx = random.sample(range(len(snack_df)), 2)
        item = snack_df.iloc[idx]
        c = item['carbohydrate'].sum(); p = item['proteins'].sum()
        f = item['fat'].sum()
        kombinasi.append(item)
        gizi.append([c, p, f])

    gizi = np.array(gizi)
    subset_idx = np.arange(len(kombinasi))
    top_k = _knn_top_k(kombinasi, gizi, target_macro, subset_idx, k)
    pemenang = _pilih_pemenang_voting(top_k)
    if pemenang is None:
        return None, None, None, []
    return pemenang['menu'], pemenang['gizi'], pemenang['jarak'], top_k


def _ringkas_top_k_untuk_tampilan(top_k, menu_terpilih_akhir):
    """
    Ubah daftar top-K (dari _knn_top_k) menjadi list of dict siap-tampil di
    template HTML: nama menu (gabungan item), kalori, jarak (error), label
    kesesuaian, dan flag `dipilih` (baris mana yang akhirnya dipilih sistem
    lewat tahap voting) — persis format Tabel 2 pada jurnal referensi.
    """
    if not top_k:
        return []
    nama_terpilih = _nama_ringkas_menu(menu_terpilih_akhir) if menu_terpilih_akhir is not None else None

    hasil = []
    for t in top_k:
        menu = t['menu']
        nama = _nama_ringkas_menu(menu)
        kalori = menu['cal'] if isinstance(menu, dict) and 'cal' in menu else float(menu['calories'].sum())
        hasil.append({
            'rank': t['rank'],
            'nama': nama,
            'kalori': round(float(kalori), 1),
            'jarak': round(float(t['jarak']), 3),
            'kesesuaian': t['kesesuaian'],
            'dipilih': (nama == nama_terpilih),
        })
    return hasil


def generate_food_recommendation_knn(target_kalori):
    mb_df    = pd.read_csv(os.path.join(BASE, 'makanan_berat_bersih.csv'))
    snack_df = pd.read_csv(os.path.join(BASE, 'snack_bersih.csv'))

    # Snack tetap difilter di level dataset (tidak dipecah per kategori)
    snack_df = filter_familiar(snack_df, WHITELIST_SNACK)

    # Makanan berat difilter PER KATEGORI di dalam _split_kategori_makanan
    kategori = _split_kategori_makanan(mb_df)

    # 1) Bangun & cari menu terbaik untuk tiap waktu makan (KNN Euclidean, terpisah)
    hasil_per_waktu = {
        waktu: _rekomendasi_satu_waktu_makan(kategori, target_kalori * rasio)
        for waktu, rasio in MEAL_KKAL_RATIO.items()
    }

    # 2) Terapkan aturan "Menu Lengkap maksimal 1x sehari", lalu susun menu
    #    final secara berurutan (Sarapan → Siang → Malam) sambil memastikan
    #    tidak ada kombinasi menu yang sama persis terpakai di lebih dari
    #    satu waktu makan.
    waktu_menu_lengkap = _tentukan_waktu_menu_lengkap(hasil_per_waktu)
    menu_final = _susun_menu_final_tanpa_duplikat(hasil_per_waktu, waktu_menu_lengkap)

    total_cal  = sum(m['menu']['cal'] for m in menu_final.values())
    total_carb = sum(m['gizi'][0]     for m in menu_final.values())
    total_prot = sum(m['gizi'][1]     for m in menu_final.values())
    total_fat  = sum(m['gizi'][2]     for m in menu_final.values())

    # 3) Snack: KNN 2 item agar mendekati target kalori camilan (15% dari total kalori harian)
    target_kalori_snack = target_kalori * SNACK_KKAL_RATIO
    target_macro_snack = [
        (target_kalori_snack * 0.55) / 4,
        (target_kalori_snack * 0.15) / 4,
        (target_kalori_snack * 0.30) / 9,
    ]
    snack_items, snack_gizi, snack_error, snack_top_k = _pilih_snack_knn(snack_df, target_macro_snack)

    # 4) Susun output siap-pakai untuk Flask/HTML
    waktu_makan_output = {}
    for waktu in ['sarapan', 'siang', 'malam']:
        pilihan = menu_final[waktu]
        icon, label = MEAL_LABEL[waktu]
        waktu_makan_output[waktu] = {
            'icon': icon,
            'label': label,
            'is_lengkap': pilihan['menu']['is_lengkap'],
            'komponen': [b.fillna('').to_dict() for b in pilihan['menu']['items']],
            'cal': pilihan['menu']['cal'],
            'target_cal': hasil_per_waktu[waktu]['target_cal'],
            'error': pilihan['error'],
            'top_k': _ringkas_top_k_untuk_tampilan(pilihan['top_k'], pilihan['menu']),
        }

    return {
        'waktu_makan': waktu_makan_output,
        'waktu_menu_lengkap': waktu_menu_lengkap,   # nama waktu makan mana yg pakai Menu Lengkap (atau None)
        'snack': snack_items.fillna('').to_dict('records'),
        'snack_error': snack_error,   # error KNN (Euclidean) khusus untuk kombinasi camilan
        'snack_top_k': _ringkas_top_k_untuk_tampilan(snack_top_k, snack_items),
        'stats': {
            'cal':  total_cal + snack_items['calories'].sum(),
            'carb': total_carb + snack_gizi[0],
            'prot': total_prot + snack_gizi[1],
            'fat':  total_fat + snack_gizi[2],
            'error': sum(m['error'] for m in menu_final.values()) + snack_error,
        },
        'target': {
            'cal':  target_kalori,
            'carb': (target_kalori * 0.55) / 4,
            'prot': (target_kalori * 0.15) / 4,
            'fat':  (target_kalori * 0.30) / 9,
            'snack_cal': target_kalori_snack,
        },
    }

# ============================================================
# 3. HTML TEMPLATE (embedded)
# ============================================================
HTML = open(os.path.join(BASE, 'template.html'), encoding='utf-8').read()

# ============================================================
# 4. ROUTES
# ============================================================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        usia  = float(request.form['usia'])
        tb    = float(request.form['tb'])
        bb    = float(request.form['bb'])
        jk    = request.form['jk']
        olahraga = request.form['olahraga']

        # FIX 6: Map UI values → training data values for tidur & screen_time
        tidur_map = {
            '< 6 jam': '< 6 jam',
            '6 - 8 jam': '6 - 8 jam',
            '> 8 jam': '> 8 jam',
        }
        screen_map = {
            '< 1 jam': '< 1 jam',
            '1 - 3 jam': '1 - 3 jam',
            '> 3 jam': '> 3 jam',
        }

        tidur_val  = tidur_map.get(request.form['tidur'], request.form['tidur'])
        screen_val = screen_map.get(request.form['screen_time'], request.form['screen_time'])

        data_baru = pd.DataFrame([{
            'Usia': usia, 'Tinggi_Badan_cm': tb, 'Berat_Badan_kg': bb,
            'Jenis_Kelamin': jk, 'daging': request.form['daging'],
            'durasi_olahraga': olahraga, 'fast_food': request.form['fast_food'],
            'tidur': tidur_val, 'screen_time': screen_val,
            'sayur_buah': request.form['sayur_buah'],
            'minuman_manis': request.form['minuman_manis'],
            'susu': request.form['susu']
        }])

        pred_gizi = model_rf.predict(data_baru)[0]
        proba = model_rf.predict_proba(data_baru)[0]
        proba_dict = dict(zip(model_rf.classes_, (proba * 100).round(1)))

        # BMR (Harris-Benedict)
        if jk == 'L':
            bmr = 66.5 + (13.7 * bb) + (5.0 * tb) - (6.8 * usia)
        else:
            bmr = 655.0 + (9.6 * bb) + (1.8 * tb) - (4.7 * usia)

        # TEE
        af_map = {'< 15 menit': 1.3, '15-30 menit': 1.56, '> 30 menit': 1.76}
        af = af_map.get(olahraga, 1.3)
        tee = bmr * af

        # ============================================================
        # KALORI TARGET + REKOMENDASI OLAHRAGA (MET = 4.0)
        # ============================================================
        MET = 4.0
        JENIS_OLAHRAGA = "Berjalan santai"
        KODE_MET = "1726260"

        # ============================================================
        # DURASI MENCAPAI BERAT BADAN IDEAL
        # BBI (Broca) = Tinggi Badan(cm) - 100
        # Asumsi perubahan berat badan = 0.5 kg/minggu (≈3500 kkal/minggu,
        # setara surplus/defisit 500 kkal/hari x 7 hari)
        # ============================================================
        def hitung_durasi_ideal(bb, tb):
            bbi = tb - 100
            selisih_bb = round(abs(bb - bbi), 1)
            durasi_minggu = selisih_bb / 0.5
            durasi_hari = round(durasi_minggu * 7, 1)
            durasi_bulan = round(durasi_minggu / 4.345, 1)
            return {
                'bbi': round(bbi, 1),
                'selisih_bb': selisih_bb,
                'durasi_minggu': round(durasi_minggu, 1),
                'durasi_hari': durasi_hari,
                'durasi_bulan': durasi_bulan,
            }

        if pred_gizi in ['Gizi Kurang']:
            total_surplus = 500
            surplus_diet  = total_surplus
            target_kalori = tee + surplus_diet
            status_kalori = "Surplus (+500 Kkal)"

            durasi_olahraga_jam   = 0.5
            durasi_olahraga_menit = 30
            kalori_olahraga = MET * bb * durasi_olahraga_jam
            durasi_ideal = hitung_durasi_ideal(bb, tb)
            rek_olahraga = {
                'mode'          : 'surplus',
                'jenis'         : JENIS_OLAHRAGA,
                'kode_met'      : KODE_MET,
                'durasi_menit'  : durasi_olahraga_menit,
                'durasi_jam'    : durasi_olahraga_jam,
                'kalori_bakar'  : round(kalori_olahraga, 1),
                'met'           : MET,
                'keterangan'    : (
                    "Olahraga ringan 30 menit direkomendasikan untuk menjaga "
                    "massa otot dan metabolisme. Kalori yang terbakar TIDAK "
                    "dikurangi dari target makan."
                ),
                **durasi_ideal,
                'target_keterangan': (
                    f"Berat badan ideal (BBI) = {tb:.0f} - 100 = {durasi_ideal['bbi']} kg. "
                    f"Selisih dari berat saat ini ({bb} kg) = {durasi_ideal['selisih_bb']} kg. "
                    f"Dengan kenaikan berat badan ~0,5 kg/minggu (surplus 500 kkal/hari), "
                    f"estimasi waktu mencapai berat ideal = {durasi_ideal['selisih_bb']} ÷ 0,5 "
                    f"= {durasi_ideal['durasi_minggu']} minggu (≈ {durasi_ideal['durasi_bulan']} bulan "
                    f"atau {durasi_ideal['durasi_hari']} hari)."
                ),
            }

        elif pred_gizi in ['Gizi Lebih', 'Obesitas']:
            total_defisit     = 500
            defisit_diet      = round(total_defisit * 0.70, 1)
            defisit_olahraga  = round(total_defisit * 0.30, 1)
            target_kalori     = tee - defisit_diet
            status_kalori     = "Defisit (-500 Kkal: 70% diet, 30% olahraga)"

            durasi_olahraga_jam   = defisit_olahraga / (MET * bb)
            durasi_olahraga_menit = round(durasi_olahraga_jam * 60, 1)
            kalori_olahraga = MET * bb * durasi_olahraga_jam
            durasi_ideal = hitung_durasi_ideal(bb, tb)
            rek_olahraga = {
                'mode'          : 'defisit',
                'jenis'         : JENIS_OLAHRAGA,
                'kode_met'      : KODE_MET,
                'defisit_diet'  : defisit_diet,
                'defisit_olah'  : defisit_olahraga,
                'durasi_menit'  : durasi_olahraga_menit,
                'durasi_jam'    : round(durasi_olahraga_jam, 3),
                'kalori_bakar'  : round(kalori_olahraga, 1),
                'met'           : MET,
                'keterangan'    : (
                    f"Defisit 500 kkal dibagi: {defisit_diet} kkal dari pengurangan "
                    f"makan (70%) + {defisit_olahraga} kkal dari olahraga (30%). "
                    f"Durasi dihitung: {defisit_olahraga} ÷ "
                    f"(MET {MET} × {bb} kg) = {durasi_olahraga_menit} menit."
                ),
                **durasi_ideal,
                'target_keterangan': (
                    f"Berat badan ideal (BBI) = {tb:.0f} - 100 = {durasi_ideal['bbi']} kg. "
                    f"Target penurunan dari berat saat ini ({bb} kg) = {durasi_ideal['selisih_bb']} kg. "
                    f"Dengan penurunan ~0,5 kg/minggu (defisit 500 kkal/hari), "
                    f"estimasi waktu mencapai berat ideal = {durasi_ideal['selisih_bb']} ÷ 0,5 "
                    f"= {durasi_ideal['durasi_minggu']} minggu (≈ {durasi_ideal['durasi_bulan']} bulan "
                    f"atau {durasi_ideal['durasi_hari']} hari)."
                ),
            }

        else:  # Gizi Baik
            target_kalori = tee
            status_kalori = "Normal (Kalori Seimbang)"
            durasi_olahraga_jam   = 0.5
            durasi_olahraga_menit = 30
            kalori_olahraga = MET * bb * durasi_olahraga_jam
            rek_olahraga = {
                'mode'          : 'normal',
                'jenis'         : JENIS_OLAHRAGA,
                'kode_met'      : KODE_MET,
                'durasi_menit'  : durasi_olahraga_menit,
                'durasi_jam'    : durasi_olahraga_jam,
                'kalori_bakar'  : round(kalori_olahraga, 1),
                'met'           : MET,
                'keterangan'    : (
                    "Pertahankan aktivitas fisik 30 menit per hari untuk menjaga "
                    "kebugaran dan berat badan ideal."
                ),
            }

        if target_kalori < 1200:
            target_kalori = 1200  # FIX 8: safety floor

        rekomendasi = generate_food_recommendation_knn(target_kalori)

        return render_template_string(HTML,
            prediksi=pred_gizi,
            proba=proba_dict,
            rek=rekomendasi,
            rek_olahraga=rek_olahraga,
            math={'bmr': round(bmr,1), 'tee': round(tee,1),
                  'target_kalori': round(target_kalori,1),
                  'status_kalori': status_kalori, 'af': af},
            form=request.form
        )

    return render_template_string(HTML, prediksi=None, rek=None, rek_olahraga=None, math=None, proba=None, form=None)

if __name__ == '__main__':
    print("✅ NutriCek Flask app starting...")
    app.run(debug=False, port=5050)