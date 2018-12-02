# README

Beispielcode für den Artikel "Unter der Haube" aus der [iX-Developer "Maschinelles Lernen"](https://shop.heise.de/zeitschriften/ix/sonderhefte) 12/2018.

Code for the article "Unter der Haube" from the german magazine iX-Developer "Maschinelles Lernen".

## Benötigt

Um das Beispiel in diesem Repository auf einem Rechner laufen zu lassen, müssen die folgenden Programme zur Verfügung stehen.

1. Ein C++-Compiler, wie z. B. von [Microsoft Visual Studio Community 2017](https://visualstudio.microsoft.com/vs/community/).
2. Die Entwicklungsumgebung [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads), bevorzugt die neueste Version (zum Zeitpunkt der Erstellung die Version 10).
3. Die Bibliothek [NVIDIA cuDNN](https://developer.nvidia.com/cudnn). Achtung, hier muss die Version zur gewählten Version des CUDA-SDKs passen. 

## Installation

Diese drei Packete sind am besten hintereinander (nicht parallel zur gleichen Zeit) zu installieren.

Es ist am einfachsten nach der Installation von CUDA die Archivdatei von cuDNN in das Verzeichnis `%CUDA_PATH%` (Windows) bzw. `$CUDA_PATH` (Linux und Mac)
zu entpacken. Die Header-Datei `cudnn.h` sollte sich anschließend mit den anderen Header-Dateien des CUDA-SDKs im gleichen Ordner befinden. Analog gilt das für die
Bibliotheken, Lib's und DLLs.

## Ausführung

Das Beispielprojekt kann mit Visual Studio Code geöffnet und ausgeführt werden.

## Überprüfung der Korrektheit

Mit den Testdaten im Projekt ergibt sich zum Beispiel die folgende Berechnung in Pseudocode:

d_y[0, 0, 3, 1] = d_image[[1,1],[1,2],[2,1],[2,2]] * d_w[[0,0],[0,1],[1,0],[1,1]] = 0 * 5 + 1 * 6 + 2 * 9 + 3 * 10 = 54

[Jörn Dinkla](https://www.dinkla.net)

(c) 2018 Jörn Dinkla, https://www.dinkla.net