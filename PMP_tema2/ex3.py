import pandas as pd

# 1. construim propriul exemplu de Paradox Simpson cu date medicale
date_medicale = {
    'Gravitate': ['Usoara', 'Usoara', 'Grava', 'Grava'],
    'Tratament': ['Nou', 'Vechi', 'Nou', 'Vechi'],
    'Pacienti_Total': [90, 10, 20, 80],
    'Pacienti_Vindecati': [81, 10, 2, 12]
}

# incarcam datele intr-un tabel (DataFrame) folosind pandas
df = pd.DataFrame(date_medicale)

# calculam rata de vindecare pentru fiecare grup specific
df['Rata_Vindecare_%'] = (df['Pacienti_Vindecati'] / df['Pacienti_Total']) * 100

print("Ratele de vindecare per categorie")
print(df[['Gravitate', 'Tratament', 'Rata_Vindecare_%']])
# aici Tratamentul Vechi pare mai bun la ambele categorii (100% > 90% si 15% > 10%)

# 2. agregam datele (ignoram gravitatea bolii) pentru a vedea inversarea
# .groupby() aduna pacientii la un loc strict in functie de Tratament
agregat = df.groupby('Tratament')[['Pacienti_Total', 'Pacienti_Vindecati']].sum().reset_index()

# calculam rata globala
agregat['Rata_Globala_%'] = (agregat['Pacienti_Vindecati'] / agregat['Pacienti_Total']) * 100

print("\n Ratele de vindecare GLOBALE (Agregate)")
print(agregat[['Tratament', 'Rata_Globala_%']])
# aici Tratamentul Nou pare mult mai bun (75.4% fata de 24.4%)