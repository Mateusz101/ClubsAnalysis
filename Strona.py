import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from Style import set_style
from ScrapingData import scrape_statistics

# Funkcja do przekształcenia kolumny w typ kategoryczny
def transform_variable_to_categorical(column, data):
    data[column] = data[column].astype('category')  # Zmiana typu danych kolumny na 'category'
    return data  # Zwrócenie zmodyfikowanego dataframe

# Funkcja zmieniająca nazwę "Premier League" na "PL" w kolumnie "Tournament"
def change_premier_league_to_pl(column, data):
    data.loc[data[column] == "Premier League", "Tournament"] = "PL"  # Zmiana wartości w kolumnie
    return data  # Zwrócenie zmodyfikowanego dataframe'u

# Funkcja do tworzenia wykresu punktowego
def plot_scatter_plot(df, x_axis, y_axis, hue=False):
    fig, ax = plt.subplots(figsize=(6, 5))  # Utworzenie figury i osi o określonym rozmiarze
    fig.patch.set_alpha(0)  # Ustawienie przezroczystości tła wykresu
    ax.set_facecolor('none')  # Ustawienie przezroczystości tła osi
    if hue:
        sns.scatterplot(data=df, x=x_axis, y=y_axis, hue="Tournament", ax=ax, palette='tab10', alpha=0.7)  # Wykres z rozróżnieniem lig
    else:
        sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax, palette='viridis', alpha=0.7)  # Wykres bez rozróżnienia
    ax.set_title(f'Scatter plot between {x_axis} and {y_axis}')  # Ustawienie tytułu wykresu
    ax.grid(True)  # Włączenie siatki na wykresie
    buf = BytesIO()  # Utworzenie bufora do zapisu wykresu
    fig.savefig(buf, format="png", transparent=True)  # Zapisanie wykresu do bufora w formacie PNG
    buf.seek(0)  # Reset pozycji odczytu w buforze
    st.image(buf)  # Wyświetlenie obrazu w aplikacji Streamlit

# Funkcja wyświetlająca typy danych w dataframe
def show_dtypes(data):
    return pd.DataFrame(data.dtypes, columns=["Dtypes"]).transpose()  # Zwrócenie dataframe z typami danych

# Funkcja wyświetlająca statystyki opisowe dataframe
def show_stats(data):
    return pd.DataFrame(data.describe())  # Zwrócenie statystyk opisowych dataframe

# Funkcja do obliczania i wizualizacji indeksu sylwetkowego dla różnych liczby klastrów
def silhouette_print(data):
    silhouette_scores = []  # Lista do przechowywania wyników silhouette
    dfn = data.select_dtypes(include=["number"])  # Wybór kolumn numerycznych
    scaler = StandardScaler()  # Inicjalizacja obiektu do standaryzacji
    dfn = scaler.fit_transform(dfn)  # Standaryzacja danych

    for i in range(2, 11):  # Iteracja przez różne liczby klastrów
        kmeans = KMeans(n_clusters=i, random_state=1)  # Inicjalizacja K-means
        cluster_labels = kmeans.fit_predict(dfn)  # Dopasowanie i predykcja klastrów
        score = silhouette_score(dfn, cluster_labels)  # Obliczanie silhouette score
        silhouette_scores.append(score)  # Dodanie wyniku do listy

    fig, ax = plt.subplots(figsize=(6, 5))  # Utworzenie figury i osi
    fig.patch.set_alpha(0)  # Przezroczystość tła figury
    ax.set_facecolor('none')  # Przezroczystość tła osi
    ax.plot(range(2, 11), silhouette_scores, marker='o', alpha=0.7)  # Rysowanie wykresu wyników indeksu silhouette
    ax.set_title('Silhouette Score')  # Tytuł wykresu
    ax.set_xlabel('Number of Clusters')  # Etykieta osi X
    ax.set_ylabel('Silhouette Score')  # Etykieta osi Y
    buf = BytesIO()  # Utworzenie bufora
    fig.savefig(buf, format="png", transparent=True)  # Zapisanie wykresu do bufora
    buf.seek(0)  # Reset pozycji odczytu
    st.image(buf)  # Wyświetlenie obrazu w aplikacji Streamlit

# Funkcja do tworzenia macierzy korelacji
def plot_correlation_matrix(df):
    # Skracanie nazw kolumn dla lepszej czytelności
    df.columns = [col[:2] + col[-2:] if len(col) > 4 else col for col in df.columns]
    corr = df.corr()  # Obliczanie macierzy korelacji
    fig, ax = plt.subplots(figsize=(6, 5))  # Utworzenie figury i osi
    fig.patch.set_alpha(0)  # Przezroczystość tła figury
    ax.set_facecolor('none')  # Przezroczystość tła osi
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, cbar=True)  # Tworzenie wykresu korelacji
    ax.set_title('Correlation Matrix')  # Tytuł wykresu
    buf = BytesIO()  # Utworzenie bufora
    fig.savefig(buf, format="png", transparent=True)  # Zapisanie wykresu do bufora
    buf.seek(0)  # Reset pozycji odczytu
    st.image(buf)  # Wyświetlenie obrazu w aplikacji Streamlit

# Funkcja do tworzenia wykresu pudełkowego
def plot_boxplot(variable_box, df, hue=True):
    fig, ax = plt.subplots(figsize=(6, 5))  # Utworzenie figury i osi
    fig.patch.set_alpha(0)  # Przezroczystość tła figury
    ax.set_facecolor('none')  # Przezroczystość tła osi
    if hue:
        sns.boxplot(data=df, y=variable_box, hue="Tournament", ax=ax)  # Wykres z podziałem na kategorie
    else:
        sns.boxplot(data=df, y=variable_box, ax=ax)  # Wykres bez podziału na kategorie
    ax.legend(framealpha=0.2)  # Dodanie legendy z przezroczystością tła
    ax.set_title(f'Box-plot for {variable_box}')  # Tytuł wykresu
    ax.grid(True)  # Włączenie siatki na wykresie
    buf = BytesIO()  # Utworzenie bufora
    fig.savefig(buf, format="png", transparent=True)  # Zapisanie wykresu do bufora
    buf.seek(0)  # Reset pozycji odczytu
    st.image(buf)  # Wyświetlenie obrazu w aplikacji Streamlit

# Funkcja do tworzenia wykresu liczb kategorii
def plot_countplot(df):
    fig, ax = plt.subplots(figsize=(6, 5))  # Utworzenie figury i osi
    fig.patch.set_alpha(0)  # Przezroczystość tła figury
    ax.set_facecolor('none')  # Przezroczystość tła osi
    sns.countplot(data=df, x="Tournament", ax=ax, palette="tab10")  # Tworzenie wykresu słupkowego
    ax.set_title(f'Countplot for specific leagues')  # Tytuł wykresu
    buf = BytesIO()  # Utworzenie bufora
    fig.savefig(buf, format="png", transparent=True)  # Zapisanie wykresu do bufora
    buf.seek(0)  # Reset pozycji odczytu
    st.image(buf)  # Wyświetlenie obrazu w aplikacji Streamlit

# Funkcja do K-means i wizualizacji wyników
def plot_kmeans_clusters(data, num_clusters, x_axis, y_axis):
    scaler = StandardScaler()  # Inicjalizacja obiektu do standaryzacji
    dfn = data.select_dtypes(include=["float", "int"])  # Wybór kolumn numerycznych
    dfn_scaled = scaler.fit_transform(dfn)  # Standaryzacja danych

    kmeans = KMeans(n_clusters=num_clusters, random_state=1)  # Inicjalizacja algorytmu K-means
    kmeans.fit(dfn_scaled)  # Dopasowanie modelu do danych

    fig, ax = plt.subplots(figsize=(6, 5))  # Utworzenie figury i osi
    fig.patch.set_alpha(0)  # Przezroczystość tła figury
    ax.set_facecolor('none')  # Przezroczystość tła osi
    sc = ax.scatter(data[x_axis], data[y_axis], c=kmeans.labels_, cmap="tab10", alpha=0.7)  # Wykres punktowy klastrów
    ax.set_title(f'K-means Clustering (k={num_clusters})')  # Tytuł wykresu
    ax.set_xlabel(x_axis)  # Etykieta osi X
    ax.set_ylabel(y_axis)  # Etykieta osi Y
    legend = ax.legend(*sc.legend_elements(), title="Clusters")  # Dodanie legendy
    ax.add_artist(legend)

    buf = BytesIO()  # Utworzenie bufora
    fig.savefig(buf, format="png", transparent=True)  # Zapisanie wykresu do bufora
    buf.seek(0)  # Reset pozycji odczytu
    st.image(buf)  # Wyświetlenie obrazu w aplikacji Streamlit

    dfn['Cluster'] = kmeans.labels_  # Dodanie klastrów jako nowej kolumny
    cluster_stats = dfn.groupby('Cluster').mean()  # Obliczenie średnich dla każdego klastra
    return cluster_stats  # Zwrócenie statystyk klastrów

# Funkcja do generowania nagłówka H1 w Streamlit
def generate_h1(text):
    st.markdown(
        f"""
            <div class='title'>  <!-- Styl dla nagłówka H1 -->
                <h1>{text}</h1>
            </div>
            """,
        unsafe_allow_html=True,
    )

# Funkcja do generowania nagłówka H2 w Streamlit
def generate_h2(text):
    st.markdown(
        f"""
            <div class='subtitle'>  <!-- Styl dla nagłówka H2 -->
                <h2>{text}</h2>
            </div>
            """,
        unsafe_allow_html=True,
    )

# Funkcja do generowania nagłówka H3 w Streamlit
def generate_h3_centered(text):
    st.markdown(
        f"""
        <div class="container">  <!-- Kontener dla wyśrodkowanego nagłówka -->
            <div class="subsubtitle">
                <h3>{text}</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Funkcja do generowania tabeli z danymi w Streamlit
def generate_head_table(df: pd.DataFrame, slides=0):
    if slides == 0:  # Jeśli liczba wierszy nie została podana, wyświetl całą tabelę
        slides = len(df)
    st.markdown(
        """
        <div class='table-container'>  <!-- Kontener dla tabeli -->
            <div class='table-content'>
                {0}
            </div>
        </div>
        """.format(df[0:slides].to_html()),  # Konwersja tabeli do formatu HTML
        unsafe_allow_html=True,
    )

# Funkcja do czyszczenia danych wejściowych
def clean_data(df_raw):
    df_raw = change_premier_league_to_pl(data=df_raw, column="Tournament")  # Zmiana nazwy ligi
    df = transform_variable_to_categorical("Tournament", df_raw)  # Konwersja zmiennych na kategoryczne
    df = df.drop_duplicates()  # Usuwanie duplikatów
    df.reset_index(drop=True, inplace=True)  # Resetowanie indeksu po usunięciu duplikatów
    return df  # Zwracanie oczyszczonego DataFrame

# Funkcja do pobierania danych w zależności od wybranej opcji
def get_data():
    # Inicjalizacja stanu, jeśli nie został jeszcze ustawiony
    if "data_source" not in st.session_state:
        st.session_state.data_source = "Use existing data"  # Domyślna opcja źródła danych
        st.session_state.data = pd.read_csv("Stare_kluby.csv")  # Wczytanie domyślnego pliku z danymi

    # Wyświetlanie przycisku do wyboru źródła danych
    scrape_decision = st.radio(
        "Choose data source:",
        options=["Use existing data", "Scrape new data"],
        index=0 if st.session_state.data_source == "Use existing data" else 1,  # Domyślnie zaznaczona opcja
    )

    # Jeśli użytkownik zmienił opcję, wykonaj odpowiednią akcję
    if scrape_decision != st.session_state.data_source:
        st.session_state.data_source = scrape_decision
        if scrape_decision == "Scrape new data":
            scrape_statistics("Clubs.csv")  # Funkcja do scrapowania
            st.session_state.data = pd.read_csv("Clubs.csv")  # Wczytanie nowego pliku
        else:
            st.session_state.data = pd.read_csv("Clubs.csv")  # Wczytanie nowego pliku (po zescrapowaniu)

    return st.session_state.data

def run_tab1(df):
    generate_h2("Overview")  # Nagłówek H2
    col1, col2 = st.columns(2)  # Generowanie dwóch kolumn
    with col1:
        generate_h3_centered("Data types")  # Nagłówek H3
        generate_head_table(show_dtypes(df))  # Tabela z typami danych
    with col2:
        generate_h3_centered("Statistics")  # Nagłówek H3
        generate_head_table(show_stats(df))  # Tabela z podstawowymi statystykami
    generate_h3_centered("The whole dataframe")  # Nagłówek H3
    rows = st.slider("Number of rows to display", 5, len(df), 5)  # Suwak do wyboru liczby wierszy do wyświetlenia
    generate_head_table(df, rows)  # Wyświetlenie wybranej liczby wierszy
    pass

def run_tab2(df):
    generate_h2("Plots")
    num_columns = df.select_dtypes(include=["number"]).columns  # Wybór tylko kolumn numerycznych
    plot_type = st.selectbox("Select plot type", ["Boxplot", "Countplot", "Scatterplot"])  # Wybór typu wykresu

    if plot_type == "Boxplot":  # Jeśli użytkownik wybrał wykres pudełkowy
        variable = st.selectbox("Variable", num_columns)  # Wybór zmiennej do wykresu pudełkowego
        hue = st.checkbox("Hue by league")  # Sprawdzenie, czy podzielić na kolory według ligi
        col1, col2 = st.columns(2)  # Tworzenie dwóch kolumn w celu podziału wykresów
        with col1:
            plot_boxplot(variable, df, hue)  # Generowanie wykresu pudełkowego
        with col2:
            plot_correlation_matrix(df.select_dtypes(include=["number"]))  # Generowanie macierzy korelacji
    elif plot_type == "Countplot":  # Jeśli użytkownik wybrał wykres słupkowy
        col1, col2 = st.columns(2)
        with col1:
            plot_countplot(df)  # Generowanie wykresu słupkowego
        with col2:
            plot_correlation_matrix(df.select_dtypes(include=["number"]))  # Generowanie macierzy korelacji
    elif plot_type == "Scatterplot":  # Jeśli użytkownik wybrał wykres rozrzutu
        x_axis = st.selectbox("X-axis", num_columns)  # Wybór zmiennej na osi X
        y_axis = st.selectbox("Y-axis", num_columns)  # Wybór zmiennej na osi Y
        hue = st.checkbox("Hue by a specific league", key=1)  # Sprawdzenie, czy podzielić na kolory według ligi
        col1, col2 = st.columns(2)
        with col1:
            plot_scatter_plot(df, x_axis, y_axis, hue)  # Generowanie wykresu rozrzutu
        with col2:
            plot_correlation_matrix(df.select_dtypes(include=["number"]))  # Generowanie macierzy korelacji

def run_tab3(df):
    generate_h2("K-means analysis")  # Nagłówek H2
    num_clusters = st.slider("Number of clusters", 2, 10, 3)  # Suwak do wyboru liczby klastrów
    x_axis = st.selectbox("X-axis", df.select_dtypes(include=["number"]).columns, key=2)  # Wybór zmiennej na osi X
    y_axis = st.selectbox("Y-axis", df.select_dtypes(include=["number"]).columns, key=3)  # Wybór zmiennej na osi Y

    col1, col2 = st.columns(2)
    with col1:
        silhouette_print(df)  # Generowanie indeksu sylwetkowego
    with col2:
        stats = plot_kmeans_clusters(df, num_clusters, x_axis, y_axis)  # Generowanie wykresu K-means

    generate_h3_centered("Means for a specific cluster")  # Nagłówek H3
    generate_head_table(stats)  # Wyświetlanie średnich dla poszczególnych klastrów

def change_variables(variable_types, df): # Zamiana wartości zmiennych w zaleznosci czy zmienna jest stymulanta czy destymulanta
    for i, var_type in enumerate(variable_types):
        if var_type == "Stimulant": # Jeśli jest stymulantą, to korzystam z funkcji stimulant, a jeśli jest destymulantą, to korzystam z funkcji destimulant
            df.iloc[:, i] = stimulant(df.iloc[:, i])
        else:
            df.iloc[:, i] = destimulant(df.iloc[:, i])

    return df # Zwracam przekształcone zmienne

def stimulant(column): # Funkcja do stymulanty
    return column # Zwracam niezmienioną kolumnę


def destimulant(column): # Funkcja do destymulanty
    return -column # Zwracam wartości o przeciwnym znaku

# Funkcja licząca dystans pomiędzy wzorcem a aktualną wartością
def distances(dataframe, maxima, weights=None):
    # Funkcja licząca dystanse pomiędzy najlepszym wzorcem
    if weights is None:
        weights = [1] * len(maxima)

    distance_frame = pd.DataFrame() # Tworzenie nowej ramki
    for i, max_value in enumerate(maxima):
        distances = weights[i] * (max_value - dataframe.iloc[:, i]) ** 2 # Dla kazdej kolumny, liczę dla kazdego wiersza odległość od wzorca
        distance_frame[dataframe.columns[i]] = distances

    return distance_frame # Zwracam ramkę odległości

# Funkcja licząca mi wartość d1 do wyniku indeksu
def di_count(distance_frame):
    di_values = distance_frame.sum(axis=1).apply(np.sqrt)
    return di_values

# Funkcja do porządkowania liniowego
def linear_ordering_Hellwig(df_original, df, weights, variable_types):
    df_changed = change_variables(variable_types, df) # Zmieniam zmienne na stymulanty/destymulanty
    clubs_transform_st = (df_changed - df_changed.mean()) / df_changed.std() # Standaryzacja
    pattern = clubs_transform_st.max(axis=0).tolist()  # Wzorce
    clubs_transform_st_dist = distances(clubs_transform_st, pattern, weights=weights)  # Odległości
    di_values = di_count(clubs_transform_st_dist)  # d1
    d0 = di_values.mean() + 2 * di_values.std()  # d0
    df_result = df_original.copy()
    df_result["HellwigIndex"] = 1 - (di_values / d0) # Indeks hellwiga
    df_result_sorted = df_result.sort_values(by="HellwigIndex", ascending=False)
    return df_result_sorted.reset_index(drop=True) # Zwracam wynik
def run_tab4(df):
    st.header("Linear Ordering") # nagłówek "Linear Ordering"
    cols = st.columns(7) # dzielę na 7 kolumn

    column_names = [
        "Goals", "Shots per Game", "Possession %", "Pass %", "Aerials Won", "Yellow Cards", "Red Cards"
    ] # nazwy kolumn

    weights = [] # wagi
    variable_types = [] # stymulanta/destymulanta

    for i, col_name in enumerate(column_names): # Tworzę dla kazdej zmiennej selectboxa oraz inputboxa
        cols[i].write(f"**{col_name}**")
        weight = cols[i].number_input("Weight", min_value=0.0, value=0.15, step=0.01, key=f"w_{col_name}")
        variable_type = cols[i].selectbox("Type", ["Stimulant", "Destimulant"], key=f"s_{col_name}") # Wartość albo "Stimulant" lub "Destimulant"
        weights.append(weight)
        variable_types.append(variable_type)

    # Example data (replace with real data)
    df_lin = df.iloc[:, [2, 3, 4, 5, 6, 8, 9]] # wybieram odpowiednie zmienne
    df_result_sorted = linear_ordering_Hellwig(df, df_lin, weights, variable_types) # wyniki metody hellwiga

    generate_h3_centered("Result")
    generate_head_table(df_result_sorted, df_result_sorted.shape[0])

# Architektura strony
def run_app():
    df_raw = get_data()  # Pobranie danych
    df = clean_data(df_raw)  # Czyszczenie danych

    generate_h1("Football Clubs Analysis")  # Generowanie nagłówka H1
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Plots", "K-means", "Linear ordering"])  # Tworzenie zakładek w interfejsie

    with tab1:  # Zakładka "Overview"
        run_tab1(df)

    with tab2:  # Zakładka "Plots"
        run_tab2(df)

    with tab3:  # Zakładka "K-means"
        run_tab3(df)

    with tab4:  # Zakładka "linear ordering"
        run_tab4(df)

# Główna funkcja aplikacji
def main():
    set_style()  # Ustawienia stylu aplikacji
    run_app() # Uruchomienie aplikacji

if __name__ == "__main__":
    main()
