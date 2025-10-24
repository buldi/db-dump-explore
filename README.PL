# Optymalizator Zrzutów SQL (db-dump-explore)

Zestaw narzędzi do optymalizacji i pracy ze standardowymi plikami zrzutów baz danych (SQL dumps). Głównym celem jest znaczne przyspieszenie procesu importowania danych do bazy, zwłaszcza w przypadku dużych plików zrzutu.

## Problem

Standardowe narzędzia takie jak `mysqldump` czy `pg_dump` często generują pliki, w których każda instrukcja `INSERT` dodaje tylko jeden lub kilka wierszy. Przy milionach rekordów, importowanie takiego pliku jest bardzo powolne, ponieważ każdy `INSERT` to osobna transakcja i narzut po stronie serwera bazy danych.

## Rozwiązanie

Ten skrypt (`optimize_sql_dump.py`) przetwarza plik zrzutu i optymalizuje go na kilka sposobów, aby import był znacznie szybszy.

## Główne Funkcje

*   **Łączenie instrukcji `INSERT`**: Skrypt łączy wiele małych instrukcji `INSERT INTO ... VALUES (...), (...), ...` w jedną, dużą instrukcję, co drastycznie redukuje liczbę zapytań do bazy.
*   **Tryb szybkiego ładowania (`--load-data`)**: Generuje pliki `.tsv` (dane rozdzielane tabulatorami) oraz plik `.sql` z instrukcją `LOAD DATA INFILE` (dla MySQL) lub `COPY` (dla PostgreSQL). Jest to najszybsza metoda importu danych.
*   **Tryb podziału (`--split`)**: Dzieli jeden duży plik zrzutu na mniejsze pliki `.sql`, po jednym dla każdej tabeli. Ułatwia to zarządzanie i importowanie tylko wybranych tabel.
*   **Automatyczne wykrywanie kompresji**: Skrypt potrafi automatycznie odczytywać skompresowane pliki (`.gz`, `.bz2`, `.xz`, `.zip`), więc nie trzeba ich ręcznie rozpakowywać.
*   **Wsparcie dla MySQL i PostgreSQL**: Automatycznie wykrywa dialekt SQL lub pozwala na jego ręczne określenie.

## Wymagania

* Python 3.x
* Opcjonalnie biblioteka `tqdm` do wyświetlania paska postępu:
  ```bash
  pip install tqdm
  ```

## Sposób Użycia

Skrypt obsługuje się z linii poleceń.

```bash
python optimize_sql_dump.py [opcje] <plik_wejściowy> [plik_wyjściowy]
```

### Przykład 1: Podstawowa optymalizacja (łączenie INSERT-ów)

Przetwarza `dump.sql.gz` i zapisuje zoptymalizowaną wersję w `dump_optimized.sql`.

```bash
python optimize_sql_dump.py --input dump.sql.gz --output dump_optimized.sql
```

### Przykład 2: Tryb podziału na tabele

Tworzy katalog `split_dump/` i umieszcza w nim osobne pliki `.sql` dla każdej tabeli ze zrzutu `big_dump.sql`.

```bash
python optimize_sql_dump.py --input big_dump.sql --split ./split_dump/
```

### Przykład 3: Tryb szybkiego ładowania (najszybsza metoda)

Tworzy katalog `fast_load/`, a w nim:
* pliki `.tsv` z danymi dla każdej tabeli,
* pliki `.sql` z instrukcjami `LOAD DATA` (MySQL) lub `COPY` (PostgreSQL) do wczytania danych z plików `.tsv`.

```bash
python optimize_sql_dump.py --input big_dump.sql --load-data ./fast_load/
```

### Przykład 4: Praca ze zrzutem PostgreSQL

Jeśli automatyczne wykrywanie zawiedzie, można jawnie określić typ bazy danych.

```bash
python optimize_sql_dump.py --db-type postgres --input pg_dump.sql --output pg_dump_optimized.sql
```

### Przykład 5: Optymalizacja tylko jednej tabeli

Przetwarza tylko instrukcje `CREATE` i `INSERT` dla tabeli `users`.

```bash
python optimize_sql_dump.py --table users --input dump.sql --output users_only.sql
```

## Wszystkie Opcje

| Opcja                 | Skrót | Opis                                                                                             |
|-----------------------|-------|--------------------------------------------------------------------------------------------------|
| `--input <plik>`      | `-i`  | Plik wejściowy zrzutu (może być skompresowany).                                                  |
| `--output <plik>`     | `-o`  | Plik wyjściowy dla zoptymalizowanego zrzutu.                                                     |
| `--db-type <typ>`     |       | Typ bazy: `mysql`, `postgres` lub `auto` (domyślnie).                                            |
| `--table <nazwa>`     | `-t`  | Optymalizuj tylko podaną tabelę.                                                                 |
| `--batch-size <liczba>`|      | Liczba wierszy w jednej połączonej instrukcji `INSERT` (domyślnie: 1000).                        |
| `--split [katalog]`   |       | Dzieli zrzut na osobne pliki per tabela w podanym katalogu (domyślnie bieżący).                  |
| `--load-data [katalog]`|      | Generuje pliki `.tsv` i `.sql` do szybkiego importu (najszybsza opcja).                          |
| `--verbose`           | `-v`  | Wyświetla dodatkowe informacje diagnostyczne i pasek postępu.                                    |
| `--dry-run`           |       | Uruchamia skrypt bez zapisywania plików wyjściowych.                                             |

## Import Równoległy

Podczas korzystania z trybu `--split` lub `--load-data`, skrypt generuje wiele plików `.sql`. Możesz importować te pliki równolegle, aby znacznie skrócić całkowity czas importu, zwłaszcza na maszynach wielordzeniowych. Świetnym narzędziem do tego celu jest GNU Parallel.

Po wygenerowaniu plików w danym katalogu (np. `fast_load/` lub `split_output/`), przejdź do tego katalogu i użyj jednego z poniższych poleceń:

### MySQL

```bash
# Opcja -j określa liczbę zadań równoległych, zazwyczaj jest to liczba rdzeni procesora.
parallel -j 8 'mysql -h <host> -u <user> -p<hasło> <nazwa_bazy> < {}' ::: *.sql
```

### PostgreSQL

```bash
# Opcja -j określa liczbę zadań równoległych.
parallel -j 8 'psql -h <host> -U <user> -d <nazwa_bazy> -f {}' ::: *.sql
```

**Uwaga:** Import równoległy jest najskuteczniejszy, gdy tabele są od siebie niezależne. Jeśli między tabelami istnieją ograniczenia klucza obcego (foreign key), może być konieczne ich wyłączenie przed importem i ponowne włączenie po jego zakończeniu lub importowanie tabel w określonej kolejności.
https://www.emu-land.net/en/computers/atari_st/tos
