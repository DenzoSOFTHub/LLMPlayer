# LLMPlayer

Motore di inferenza LLM in puro Java per eseguire modelli GGUF in locale. Zero dipendenze esterne — utilizza solo il JDK. Supporta le architetture Llama, Qwen2, Qwen3, DeepSeek2 e GLM4 con formati quantizzati (Q2_K, Q3_K, Q4_0, Q4_K, Q5_0, Q5_K, Q6_K, Q8_0, BF16, F16, F32).

## Requisiti

- **Java 8** — funzionalita' base (niente SIMD, niente GPU)
- **Java 21+** — aggiunge SIMD (Vector API), memory mapping ottimizzato (Panama FFI), accelerazione GPU (OpenCL)
- **Java 25** — aggiunge parallelismo avanzato (StructuredTaskScope, virtual thread matmul)
- **Maven 3.x** — per la compilazione
- **Driver OpenCL** — solo se si vuole usare la GPU (es. `libOpenCL.so` su Linux)

## Compilazione

Il progetto ha tre profili Maven che includono set diversi di sorgenti:

```bash
# Profilo java25 (default) — include tutto: java/ + java21/ + java25/
mvn clean compile

# Profilo java21 — include java/ + java21/ (no StructuredTaskScope, no virtual thread matmul)
mvn clean compile -Pjava21

# Profilo java8 — include solo java/ (no Vector API, no Panama FFI, no GPU)
mvn clean compile -Pjava8
```

## Esecuzione

### Con Java 8

Dopo aver compilato con `-Pjava8`:

```bash
java -cp target/classes it.denzosoft.llmplayer.LLMPlayer [opzioni]
```

Nessun flag JVM aggiuntivo necessario. Non sono disponibili l'accelerazione SIMD, la GPU e le ottimizzazioni di memory mapping — il sistema usa `MappedByteBuffer` e operazioni scalari come fallback.

### Con Java 21+

Dopo aver compilato con `-Pjava21`:

```bash
java --add-modules jdk.incubator.vector \
     --enable-native-access=ALL-UNNAMED \
     -cp target/classes \
     it.denzosoft.llmplayer.LLMPlayer [opzioni]
```

I tre flag JVM sono obbligatori:
- `--add-modules jdk.incubator.vector` — abilita la Vector API per le operazioni SIMD sui tensori
- `--enable-native-access=ALL-UNNAMED` — abilita Panama FFI per il memory mapping e i binding OpenCL

### Con Java 25

Dopo aver compilato con il profilo default:

```bash
java --add-modules jdk.incubator.vector \
     --enable-native-access=ALL-UNNAMED \
     --enable-preview \
     -cp target/classes \
     it.denzosoft.llmplayer.LLMPlayer [opzioni]
```

Il flag `--enable-preview` abilita `StructuredTaskScope` per la generazione batch parallela e i virtual thread per il matmul multi-thread.

### Script di avvio (Java 25)

Per comodita', sono disponibili script pre-configurati con tutti i flag:

```bash
# Linux / macOS
./run.sh [opzioni]

# Windows
run.bat [opzioni]
```

## Modalita' di utilizzo

### GUI Desktop (default)

Lanciare senza argomenti per aprire l'interfaccia Swing:

```bash
./run.sh
```

La GUI permette di:
- Selezionare un modello GGUF dalla directory (default: `gguf/`)
- Configurare temperatura, top-k, top-p, repetition penalty, contesto e max token
- Chat interattiva con streaming token-by-token
- Monitoraggio in tempo reale di CPU e RAM
- Avviare/fermare il web server integrato

### CLI — Prompt singolo

```bash
./run.sh --model percorso/modello.gguf --prompt "Spiega cos'e' l'intelligenza artificiale" --max-tokens 512
```

L'output viene stampato in streaming token per token. Al termine vengono mostrate le statistiche (token generati, velocita', tempo).

### CLI — Chat interattiva

```bash
./run.sh --model percorso/modello.gguf --interactive
```

Digitare i messaggi e premere Invio. Comandi speciali: `quit` o `exit` per uscire, `info` per i dettagli del modello.

### Web UI

```bash
./run.sh --web --port 8080 --gguf-dir ./modelli
```

Apre un server HTTP su `http://localhost:8080` con un'interfaccia web e API REST. La porta e' configurabile (default: 8080).

### Informazioni modello

```bash
./run.sh --model percorso/modello.gguf --info
```

Stampa i metadati del modello (architettura, layer, dimensioni, vocabolario) e termina.

## Accelerazione GPU (OpenCL)

Richiede Java 21+ e driver OpenCL installati.

### Elencare i dispositivi disponibili

```bash
./run.sh --gpu-list
```

Mostra tutti i dispositivi OpenCL rilevati (GPU, CPU OpenCL, acceleratori) con nome e memoria. Se non vengono trovati dispositivi, verificare che i driver OpenCL siano installati (`libOpenCL.so` su Linux, driver GPU del produttore su Windows/macOS).

### Abilitare la GPU

```bash
# Usa il primo dispositivo GPU (device 0)
./run.sh --model modello.gguf --gpu --prompt "Ciao" --max-tokens 256

# Usa un dispositivo specifico
./run.sh --model modello.gguf --gpu-device 1 --prompt "Ciao" --max-tokens 256
```

Il flag `--gpu-device N` seleziona il dispositivo per indice (come mostrato da `--gpu-list`) e abilita automaticamente la GPU. Il flag `--gpu` da solo usa il device 0.

### Come funziona

Quando la GPU e' abilitata, il sistema:
1. Inizializza un contesto OpenCL sul dispositivo selezionato
2. Registra un `GpuBufferManager` globale nel `TensorFactory`
3. Per ogni tensore creato durante il caricamento del modello, tenta prima la variante GPU (es. `Q4_KGpuTensor`), poi ricade sulla variante CPU se non disponibile
4. Le operazioni pesanti (matmul, RMSNorm, softmax, SiLU, ecc.) vengono eseguite tramite kernel OpenCL compilati on-demand

I formati quantizzati supportati su GPU sono: F32, Q4_0, Q4_K, Q5_K, Q6_K, Q8_0. Gli altri formati ricadono automaticamente sulla CPU.

Se Java e' < 21 o i driver OpenCL non sono presenti, il sistema stampa un avviso e prosegue in modalita' solo CPU.

## Cosa cambia tra i profili Java

| Funzionalita' | Java 8 | Java 21 | Java 25 |
|---|:---:|:---:|:---:|
| Inferenza base (tutte le architetture) | Si | Si | Si |
| GUI Desktop (Swing) | Si | Si | Si |
| Web Server | Si | Si | Si |
| Tutti i formati di quantizzazione (CPU) | Si | Si | Si |
| Operazioni tensore SIMD (Vector API) | No | Si | Si |
| Memory mapping ottimizzato (Panama FFI) | No | Si | Si |
| Accelerazione GPU (OpenCL) | No | Si | Si |
| Virtual thread matmul | No | No | Si |
| Generazione batch parallela (StructuredTaskScope) | No | No | Si |

Il degradamento e' automatico: le classi Java 21/25 vengono caricate via riflessione (`Class.forName`). Se non sono disponibili, il sistema usa i fallback Java 8 (operazioni scalari, `MappedByteBuffer`, thread pool standard) senza errori.

## Opzioni CLI

| Opzione | Alias | Tipo | Default | Descrizione |
|---|---|---|---|---|
| `--model` | `-m` | String | — | Percorso del file GGUF |
| `--prompt` | `-p` | String | — | Prompt per generazione singola |
| `--interactive` | `-i` | Flag | false | Modalita' chat interattiva |
| `--max-tokens` | `-n` | Intero | 256 | Numero massimo di token da generare |
| `--temperature` | `-t` | Float | 0.7 | Temperatura di campionamento (0 = deterministico, >1 = piu' casuale) |
| `--top-k` | — | Intero | 40 | Mantiene solo i K token piu' probabili |
| `--top-p` | — | Float | 0.9 | Nucleus sampling: soglia di probabilita' cumulativa |
| `--repetition-penalty` | — | Float | 1.1 | Penalita' per token ripetuti (>1 riduce le ripetizioni) |
| `--seed` | — | Long | casuale | Seed per riproducibilita' |
| `--threads` | — | Intero | num CPU | Numero di thread di lavoro |
| `--context-length` | `-c` | Intero | 2048 | Lunghezza massima del contesto (token) |
| `--info` | — | Flag | false | Mostra info modello e termina |
| `--web` | `-w` | Flag | false | Avvia il server web |
| `--port` | — | Intero | 8080 | Porta del server web |
| `--gguf-dir` | — | String | `gguf` | Directory dei file GGUF |
| `--gpu` | — | Flag | false | Abilita accelerazione GPU |
| `--gpu-device` | — | Intero | 0 | Indice del dispositivo GPU |
| `--gpu-list` | — | Flag | false | Elenca i dispositivi GPU e termina |
| `--help` | `-h` | Flag | false | Mostra l'aiuto |

## API REST (modalita' web)

Quando si avvia con `--web`, il server espone le seguenti API:

### Modelli

| Endpoint | Metodo | Descrizione |
|---|---|---|
| `/api/models` | GET | Elenco dei file GGUF nella directory |
| `/api/models/load` | POST | Carica un modello: `{"path": "...", "contextLength": 2048}` |
| `/api/models/unload` | POST | Scarica il modello corrente |
| `/api/models/info` | GET | Metadati del modello caricato |

### Chat

| Endpoint | Metodo | Descrizione |
|---|---|---|
| `/api/chat` | POST | Generazione con streaming (Server-Sent Events) |
| `/api/chat/stop` | POST | Interrompe la generazione in corso |

Richiesta `/api/chat`:
```json
{
  "prompt": "Il tuo messaggio",
  "systemMessage": "Messaggio di sistema opzionale",
  "temperature": 0.7,
  "maxTokens": 256,
  "topK": 40,
  "topP": 0.9,
  "repPenalty": 1.1
}
```

La risposta e' uno stream SSE:
```
data: {"token": "ciao", "done": false}
data: {"token": " mondo", "done": false}
data: {"done": true, "stats": {"tokenCount": 10, "promptTokenCount": 5, "tokensPerSecond": 25.5, "timeMs": 392}}
```

## API Java

```java
// Caricamento
LLMEngine engine = LLMEngine.load(Path.of("modello.gguf"), 2048);

// Generazione singola
GenerationResponse resp = engine.generate(
    GenerationRequest.builder()
        .prompt("Ciao, come stai?")
        .maxTokens(256)
        .samplerConfig(SamplerConfig.builder()
            .temperature(0.7f)
            .topK(40)
            .topP(0.9f)
            .repetitionPenalty(1.1f)
            .build())
        .build()
);
System.out.println(resp.text());

// Generazione con streaming
engine.generate(request, (token, id) -> {
    System.out.print(token);
    return true;  // restituire false per interrompere
});

// Generazione batch (usa StructuredTaskScope su Java 25, thread pool altrimenti)
List<GenerationResponse> risposte = engine.generateBatch(listaRichieste);

// Caricamento con GPU
GpuConfig gpu = new GpuConfig();
gpu.setEnabled(true);
gpu.setDeviceId(0);
LLMEngine engineGpu = LLMEngine.load(Path.of("modello.gguf"), 2048, gpu);

// Pulizia
engine.close();
```

`LLMEngine` e' thread-safe: i pesi del modello sono immutabili (memory-mapped) e ogni chiamata a `generate()` crea il proprio stato di inferenza.

## Architetture supportate

| Architettura | Chiave GGUF | Tokenizer | Template chat |
|---|---|---|---|
| Llama (1/2/3) | `llama` | BPE (gpt2) / SentencePiece | `<\|start_header_id\|>user<\|end_header_id\|>` |
| Qwen2 | `qwen2` | BPE (gpt2) | `<\|im_start\|>user` |
| Qwen3 | `qwen3` | BPE (gpt2) | `<\|im_start\|>user` |
| DeepSeek2 | `deepseek2` | BPE (gpt2) | `User: ... Assistant:` |
| GLM4 | `glm4` | SentencePiece | `[gMASK]<sop><\|user\|>` |

L'architettura viene rilevata automaticamente dal campo `general.architecture` nei metadati GGUF.
