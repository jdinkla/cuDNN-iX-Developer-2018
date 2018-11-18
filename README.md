# README

Beispielcode für den Artikel "Unter der Haube" aus der ix-Developer "Maschinelles Lernen" 12/2018.

## Benötigt

Um das Beispiel in diesem Repository auf einem Rechner laufen zu lassen, müssen die folgenden Programme zur Verfügung stehen.

1. Ein C++-Compiler, wie z. B. von Microsoft Visual Studio Community 2017.
2. Die Entwicklungsumgebung [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads), bevorzugt die neueste Version (zum Zeitpunkt der Erstellung die Version 10).
3. Die Bibliothek [NVIDIA cuDNN](https://developer.nvidia.com/cudnn). Achtung, hier muss die Version zur gewählten Version des CUDA-SDKs passen. 

## Installation

Diese drei Packete sind am besten hintereinander (nicht parallel zur gleichen Zeit) zu installieren.

Es ist am einfachsten nach der Installation von CUDA die Archivdatei von cuDNN in das Verzeichnis `%CUDA_PATH%` (Windows) bzw. `$CUDA_PATH` (Linux und Mac)
zu entpacken. Die Header-Datei `cudnn.h` sollte sich anschließend mit den anderen Header-Dateien des CUDA-SDKs im gleichen Ordner befinden. Analog gilt das für die
Bibliotheken, Lib's und DLLs.

## Ausführung

Das Beispielprojekt kann mit Visual Studio Code geöffnet und ausgeführt werden.

```bash
TODO Beispiel
```

[Jörn Dinkla](https://www.dinkla.net)