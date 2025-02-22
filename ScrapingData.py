from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException, ElementClickInterceptedException  # Obsługa wyjątków
import pandas as pd
from bs4 import BeautifulSoup

def setup_driver(url):
    driver = webdriver.Chrome()  # Tworzenie instancji przeglądarki Chrome
    driver.get(url)  # Otwieranie strony internetowej z podanym URL
    return driver  # Zwracanie instancji WebDriver

def extract_all_data(driver):
    WebDriverWait(driver, 15).until(  # Czeka aż element z klasą 'stat-table' stanie się widoczny
        EC.presence_of_element_located((By.CLASS_NAME, "stat-table"))
    )

    headers = []  # Inicjalizacja listy nagłówków tabeli
    data = []  # Inicjalizacja listy na dane z tabeli

    while True:
        try:
            WebDriverWait(driver, 10).until(  # Czeka, aż tabela będzie dostępna
                EC.presence_of_element_located((By.CLASS_NAME, "stat-table"))
            )
            table = driver.find_element(By.CLASS_NAME, "stat-table")  # Znajdowanie tabeli na stronie

            rows = table.find_elements(By.TAG_NAME, "tr")  # Pobieranie wszystkich wierszy tabeli
            if not headers:  # Pobieramy nagłówki tylko raz (pierwszy wiersz)
                headers = [header.text for header in rows[0].find_elements(By.TAG_NAME, "th")]

            for row in rows[1:]:  # Pomijamy nagłówki
                cols = row.find_elements(By.TAG_NAME, "td")  # Pobieramy komórki wiersza
                row_data = [col.text for col in cols]  # Dodajemy dane do wiersza

                # Wyciąganie danych z 'Discipline' (żółte i czerwone kartki)
                discipline = cols[4].get_attribute("outerHTML")  # Pobieramy HTML z 'Discipline'
                yellow_cards, red_cards = extract_cards(discipline)  # Wyciąganie kartek

                # Dodanie danych o kartkach do wiersza
                row_data.extend([yellow_cards, red_cards])
                data.append(row_data)  # Dodanie wiersza

            WebDriverWait(driver, 10).until(  # Czekaj aż przycisk 'next' będzie klikalny
                EC.element_to_be_clickable((By.ID, "next"))
            )
            next_button = driver.find_element(By.ID, "next")  # Znajdowanie przycisku 'next'

            if "disabled" in next_button.get_attribute("class"):  # Sprawdzanie, czy przycisk jest aktywny
                break

            next_button.click()  # Kliknięcie przycisku 'next', aby przejść do następnej strony

        except ElementClickInterceptedException:  # Obsługuje błąd, gdy kliknięcie elementu jest zablokowane
            print("Element click intercepted - ponawiam")
            continue  # Powtarzamy próbę kliknięcia

        except StaleElementReferenceException:  # Obsługuje błąd, gdy element staje się nieaktualny
            print("Element stał się nieaktualny, ponawiam")
            continue  # Ponowna próba lokalizacji elementu

        except TimeoutException:  # Obsługuje błąd, gdy czas oczekiwania przekroczy limit
            print("Czas przekroczony.")
            break  # Kończenie pętli, gdy nie ma kolejnej strony

    return headers + ["Yellow Cards", "Red Cards"], data  # Zwracamy nagłówki i dane z dodatkowymi kolumnami

def extract_cards(discipline):
    yellow_cards = 0  # Domyślna liczba żółtych kartek
    red_cards = 0  # Domyślna liczba czerwonych kartek
    try:
        if pd.notnull(discipline):  # Sprawdzamy, czy 'discipline' nie jest nullem
            soup = BeautifulSoup(discipline, 'html.parser')  # Parsowanie HTML za pomocą BeautifulSoup
            yellow_cards = int(soup.find('span', class_='yellow-card-box').text) if soup.find('span', class_='yellow-card-box') else 0 # Wyciąganie informacji o żółtych kartkach
            red_cards = int(soup.find('span', class_='red-card-box').text) if soup.find('span', class_='red-card-box') else 0 # Wyciąganie informacji o czerwonych kartkach
    except Exception as e:
        print(f"Błąd z discipline: {e}")
    return yellow_cards, red_cards  # Zwracamy liczbę żółtych i czerwonych kartek

def clean_data(dataframe):
    if "Team" in dataframe.columns:  # Sprawdzamy, czy kolumna 'Team' istnieje
        dataframe["Team"] = dataframe["Team"].str.replace(r"^\d+\.\s*", "", regex=True)  # Usuwamy numerację drużyn
    df = dataframe.drop(["Discipline"], axis=1)  # Usuwamy kolumnę 'Discipline', bo nie jest już potrzebna
    return df  # Zwracamy oczyszczony DataFrame

def scrape_statistics(file_name): # Główna funkcja
    url = "http://whoscored.com/Statistics"  # URL strony
    driver = setup_driver(url)  # Inicjalizowanie WebDrivera i otwieranie strony

    try:
        headers, data = extract_all_data(driver)  # Wyciąganie danych
        driver.quit()  # Zamykanie WebDrivera po zakończeniu
        df = pd.DataFrame(data=data, columns=headers)  # Tworzymy DataFrame z danych
        df_cleaned = clean_data(df)  # Oczyszczanie danych
        df_cleaned.to_csv(path_or_buf=file_name, index=False)  # Zapisujemy oczyszczone dane do pliku CSV
        print(f"Dane zapisano do pliku: {file_name}")  # Informacja o zapisaniu danych

    except Exception as e:
        print(f"Błąd: {e}")
        driver.quit()
