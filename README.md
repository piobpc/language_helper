Pomocnik Językowy (v9)

_ _ _ _  README _ _ _ _   

Drobne zmiany względem oryginalnej specyfikacji: 

1 __ dodatkowe importy, aby umożliwić efektywne wykonanie kodu: 
    A - from typing import List 
    B - import uuid (QDrant)
    C - from bs4 import BeautifulSoup (usuwanie html ze stringów dla audio generatora)

2 __ dodatkowy kod wygenerowany przy pomocy Chata GPT, oznaczony odpowiednio komentarzami
jako dodatkowa pomoc edukacyjna (momentami kod wykracza mocno poza zakres kursu):
    A - usprawnienie analiz tekstowych
    B - usprawniony wizualnie interfejs/czcionki (użycie html)
    C - usprawnienie bazy danych 

Co można jeszcze usprawnić w kolejnych wersjach?
1 __ generowanie audio - obecny w kodzie model tts sprawnie odczytuje wskazany tekst,
natomiast nie zawsze radzi sobie z odróżnianiem słów w jednym tekście w różnych językach 
(skutek: w tłumaczeniu części zdań niektóre słowa angielskie czyta z wymową po polsku itd.)
2 __ dodanie możliwości edycji notatek, które później byłyby zamieniane na audio
3 __ dodanie możliwości zapisu notatek audio 
   