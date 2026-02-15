package it.denzosoft.llmplayer.ui;

import it.denzosoft.llmplayer.api.*;
import it.denzosoft.llmplayer.gguf.GGUFFile;
import it.denzosoft.llmplayer.gguf.GGUFParser;
import it.denzosoft.llmplayer.gpu.GpuConfig;
import it.denzosoft.llmplayer.sampler.SamplerConfig;
import it.denzosoft.llmplayer.web.WebServer;

import javax.swing.*;
import javax.swing.text.*;
import java.awt.*;
import java.awt.event.*;
import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.List;

/**
 * Swing desktop UI for LLMPlayer.
 * Simple system-native look. Sidebar with controls, chat area with streaming.
 */
public class LLMPlayerUI extends JFrame {

    private final String ggufDirectory;
    private volatile LLMEngine engine;
    private volatile boolean generating;
    private volatile boolean stopRequested;
    private WebServer webServer;

    // Sidebar
    private JComboBox<ModelEntry> modelCombo;
    private JButton loadBtn, unloadBtn;
    private JLabel statusLabel;
    private JLabel infoName, infoArch, infoLayers, infoHeads, infoEmbed, infoCtx, infoVocab, infoLoad;
    private JLabel infoModelSize, infoKvCache, infoTotalRam;
    private JLabel infoGpuDevice, infoGpuLayers;
    private PerformanceMonitor perfMonitor;
    private JCheckBox gpuCheckbox;
    private JComboBox<String> gpuDeviceCombo;
    private JSpinner gpuLayersSpinner;
    private List<Map<String, Object>> gpuDeviceList = new ArrayList<>();
    private JSpinner tempSpinner, maxTokensSpinner, topKSpinner, topPSpinner, repPenaltySpinner, ctxSpinner;
    private JTextArea systemMsgArea;
    private JSpinner portSpinner;
    private JButton webStartBtn, webStopBtn;
    private JLabel webStatusLabel;

    // Chat
    private JTextPane chatPane;
    private StyledDocument chatDoc;
    private JTextArea inputArea;
    private JButton sendBtn;

    private static final Set<String> SUPPORTED_ARCHS;
    static {
        Set<String> set = new HashSet<>();
        set.add("llama");
        set.add("qwen2");
        set.add("qwen3");
        set.add("glm4");
        set.add("deepseek2");
        SUPPORTED_ARCHS = Collections.unmodifiableSet(set);
    }

    static final class ModelEntry {
        private final String name;
        private final String path;
        private final long size;
        private final String arch;
        private final boolean supported;

        ModelEntry(String name, String path, long size, String arch, boolean supported) {
            this.name = name;
            this.path = path;
            this.size = size;
            this.arch = arch;
            this.supported = supported;
        }

        String name() { return name; }
        String path() { return path; }
        long size() { return size; }
        String arch() { return arch; }
        boolean supported() { return supported; }

        @Override public String toString() {
            String label = name + " (" + (size / 1024 / 1024) + " MB)";
            if (!supported) label += " [" + arch + " - unsupported]";
            return label;
        }
    }

    public LLMPlayerUI(String ggufDirectory) {
        super("LLMPlayer");
        this.ggufDirectory = ggufDirectory;
        try { UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName()); }
        catch (Exception ignored) {}
        buildFrame();
        loadModelList();
    }

    public static void launch(String ggufDirectory) {
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                new LLMPlayerUI(ggufDirectory).setVisible(true);
            }
        });
    }

    // -- Frame --

    private void buildFrame() {
        setSize(1050, 700);
        setMinimumSize(new Dimension(750, 450));
        setLocationRelativeTo(null);
        setDefaultCloseOperation(DO_NOTHING_ON_CLOSE);
        addWindowListener(new WindowAdapter() {
            @Override public void windowClosing(WindowEvent e) {
                shutdown(); dispose(); System.exit(0);
            }
        });

        JScrollPane sidebarScroll = new JScrollPane(buildSidebar(),
            JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED, JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);
        sidebarScroll.setPreferredSize(new Dimension(290, 0));

        JSplitPane split = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, sidebarScroll, buildChatArea());
        split.setDividerLocation(290);
        add(split);
    }

    // -- Sidebar (GridBagLayout) --

    private JPanel buildSidebar() {
        JPanel p = new ScrollablePanel(new GridBagLayout());
        p.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        GridBagConstraints gc = new GridBagConstraints();
        gc.anchor = GridBagConstraints.WEST;
        gc.fill = GridBagConstraints.HORIZONTAL;
        gc.gridx = 0; gc.gridwidth = 2; gc.weightx = 1;
        int row = 0;

        // Title
        JLabel title = new JLabel("LLMPlayer v1.0");
        title.setFont(title.getFont().deriveFont(Font.BOLD, 16f));
        gc.gridy = row++; gc.insets = new Insets(0, 0, 10, 0);
        p.add(title, gc);

        // -- Model --
        gc.gridy = row++; gc.insets = new Insets(0, 0, 4, 0);
        p.add(sectionLabel("Model"), gc);

        modelCombo = new JComboBox<>();
        modelCombo.setRenderer(new DefaultListCellRenderer() {
            @Override public Component getListCellRendererComponent(JList<?> list, Object value,
                    int index, boolean sel, boolean focus) {
                super.getListCellRendererComponent(list, value, index, sel, focus);
                if (value instanceof ModelEntry && !((ModelEntry) value).supported()) {
                    setForeground(Color.GRAY);
                }
                return this;
            }
        });
        gc.gridy = row++; gc.insets = new Insets(0, 0, 4, 0);
        p.add(modelCombo, gc);

        JPanel btnRow = new JPanel(new GridLayout(1, 2, 4, 0));
        loadBtn = new JButton("Load");
        loadBtn.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                loadModel();
            }
        });
        unloadBtn = new JButton("Unload");
        unloadBtn.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                unloadModel();
            }
        });
        unloadBtn.setEnabled(false);
        btnRow.add(loadBtn);
        btnRow.add(unloadBtn);
        gc.gridy = row++; gc.insets = new Insets(0, 0, 4, 0);
        p.add(btnRow, gc);

        statusLabel = new JLabel("No model loaded");
        gc.gridy = row++; gc.insets = new Insets(0, 0, 10, 0);
        p.add(statusLabel, gc);

        // -- Model Info --
        gc.gridy = row++; gc.insets = new Insets(0, 0, 2, 0);
        p.add(sectionLabel("Model Info"), gc);

        String[] labels = {"Name", "Arch", "Layers", "Heads", "Embedding", "Context", "Vocab", "Load time"};
        JLabel[] values = new JLabel[8];
        for (int i = 0; i < labels.length; i++) {
            values[i] = new JLabel("\u2014");
            row = addRow(p, gc, row, labels[i], values[i]);
        }
        infoName = values[0]; infoArch = values[1]; infoLayers = values[2]; infoHeads = values[3];
        infoEmbed = values[4]; infoCtx = values[5]; infoVocab = values[6]; infoLoad = values[7];

        // Memory estimate rows
        infoModelSize = new JLabel("\u2014");
        row = addRow(p, gc, row, "Model Size", infoModelSize);
        infoKvCache = new JLabel("\u2014");
        row = addRow(p, gc, row, "KV Cache", infoKvCache);
        infoTotalRam = new JLabel("\u2014");
        row = addRow(p, gc, row, "Total RAM", infoTotalRam);
        infoGpuDevice = new JLabel("\u2014");
        row = addRow(p, gc, row, "GPU", infoGpuDevice);
        infoGpuLayers = new JLabel("\u2014");
        row = addRow(p, gc, row, "GPU Layers", infoGpuLayers);

        // spacer
        gc.gridx = 0; gc.gridwidth = 2; gc.gridy = row++;
        gc.insets = new Insets(6, 0, 0, 0);
        p.add(new JSeparator(), gc);

        // -- Performance Monitor --
        gc.gridy = row++; gc.insets = new Insets(6, 0, 4, 0);
        p.add(sectionLabel("Performance"), gc);

        perfMonitor = new PerformanceMonitor();
        gc.gridy = row++; gc.insets = new Insets(0, 0, 0, 0);
        gc.fill = GridBagConstraints.BOTH;
        p.add(perfMonitor, gc);
        gc.fill = GridBagConstraints.HORIZONTAL;
        perfMonitor.start();

        // spacer
        gc.gridx = 0; gc.gridwidth = 2; gc.gridy = row++;
        gc.insets = new Insets(6, 0, 0, 0);
        p.add(new JSeparator(), gc);

        // -- Services --
        gc.gridy = row++; gc.insets = new Insets(6, 0, 4, 0);
        p.add(sectionLabel("Services"), gc);

        portSpinner = new JSpinner(new SpinnerNumberModel(8080, 1024, 65535, 1));
        row = addRow(p, gc, row, "Web API Port", portSpinner);

        JPanel webBtnRow = new JPanel(new GridLayout(1, 2, 4, 0));
        webStartBtn = new JButton("Start");
        webStartBtn.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                startWebServer();
            }
        });
        webStopBtn = new JButton("Stop");
        webStopBtn.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                stopWebServer();
            }
        });
        webStopBtn.setEnabled(false);
        webBtnRow.add(webStartBtn);
        webBtnRow.add(webStopBtn);
        gc.gridx = 0; gc.gridwidth = 2; gc.gridy = row++;
        gc.insets = new Insets(0, 0, 2, 0);
        p.add(webBtnRow, gc);

        webStatusLabel = new JLabel("Not running");
        gc.gridy = row++; gc.insets = new Insets(0, 0, 6, 0);
        p.add(webStatusLabel, gc);

        gc.gridy = row++;
        p.add(new JSeparator(), gc);

        // -- Generation Settings --
        gc.gridy = row++; gc.insets = new Insets(6, 0, 4, 0);
        p.add(sectionLabel("Generation Settings"), gc);

        tempSpinner = new JSpinner(new SpinnerNumberModel(0.7, 0.0, 2.0, 0.1));
        row = addRow(p, gc, row, "Temperature", tempSpinner);
        maxTokensSpinner = new JSpinner(new SpinnerNumberModel(2048, 1, 131072, 64));
        row = addRow(p, gc, row, "Max Tokens", maxTokensSpinner);
        topKSpinner = new JSpinner(new SpinnerNumberModel(40, 1, 200, 1));
        row = addRow(p, gc, row, "Top-K", topKSpinner);
        topPSpinner = new JSpinner(new SpinnerNumberModel(0.9, 0.0, 1.0, 0.05));
        row = addRow(p, gc, row, "Top-P", topPSpinner);
        repPenaltySpinner = new JSpinner(new SpinnerNumberModel(1.1, 1.0, 2.0, 0.05));
        row = addRow(p, gc, row, "Rep. Penalty", repPenaltySpinner);
        ctxSpinner = new JSpinner(new SpinnerNumberModel(2048, 512, 131072, 256));
        row = addRow(p, gc, row, "Context Len.", ctxSpinner);

        // -- GPU --
        gc.gridx = 0; gc.gridwidth = 2; gc.gridy = row++;
        gc.insets = new Insets(6, 0, 0, 0);
        p.add(new JSeparator(), gc);

        gc.gridy = row++; gc.insets = new Insets(6, 0, 4, 0);
        p.add(sectionLabel("GPU"), gc);

        gpuCheckbox = new JCheckBox("Enable GPU");
        gpuCheckbox.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                boolean sel = gpuCheckbox.isSelected();
                gpuDeviceCombo.setEnabled(sel);
                gpuLayersSpinner.setEnabled(sel);
            }
        });
        gc.gridy = row++; gc.insets = new Insets(0, 0, 4, 0);
        p.add(gpuCheckbox, gc);

        gpuDeviceCombo = new JComboBox<>();
        gpuDeviceCombo.setEnabled(false);
        row = addRow(p, gc, row, "Device", gpuDeviceCombo);

        gpuLayersSpinner = new JSpinner(new SpinnerNumberModel(-1, -1, 999, 1));
        gpuLayersSpinner.setEnabled(false);
        gpuLayersSpinner.setToolTipText("-1 = Auto, 0 = All layers, N = first N layers on GPU");
        row = addRow(p, gc, row, "GPU Layers", gpuLayersSpinner);

        // Populate GPU devices in background, auto-enable best device
        new Thread(new Runnable() {
            @Override
            public void run() {
                final List<Map<String, Object>> devices = LLMEngine.listGpuDevices();
                SwingUtilities.invokeLater(new Runnable() {
                    @Override
                    public void run() {
                        gpuDeviceList = devices;
                        gpuDeviceCombo.removeAllItems();
                        if (devices.isEmpty()) {
                            gpuDeviceCombo.addItem("No GPU devices found");
                            gpuCheckbox.setEnabled(false);
                        } else {
                            int bestIdx = 0;
                            long bestVram = 0;
                            for (int i = 0; i < devices.size(); i++) {
                                Map<String, Object> dev = devices.get(i);
                                String name = String.valueOf(dev.get("name"));
                                Object mem = dev.get("globalMemory");
                                long vram = 0;
                                if (mem instanceof Number) {
                                    vram = ((Number) mem).longValue();
                                    long mb = vram / (1024 * 1024);
                                    name = name + " (" + mb + " MB)";
                                }
                                gpuDeviceCombo.addItem(name);
                                if (vram > bestVram) {
                                    bestVram = vram;
                                    bestIdx = i;
                                }
                            }
                            // Auto-enable GPU with best device selected
                            gpuCheckbox.setSelected(true);
                            gpuDeviceCombo.setSelectedIndex(bestIdx);
                            gpuDeviceCombo.setEnabled(true);
                            gpuLayersSpinner.setEnabled(true);
                        }
                    }
                });
            }
        }, "gpu-enumerate").start();

        // -- System Message --
        gc.gridx = 0; gc.gridwidth = 2; gc.gridy = row++;
        gc.insets = new Insets(6, 0, 2, 0);
        p.add(sectionLabel("System Message"), gc);

        systemMsgArea = new JTextArea(3, 10);
        systemMsgArea.setLineWrap(true);
        systemMsgArea.setWrapStyleWord(true);
        JScrollPane sysScroll = new JScrollPane(systemMsgArea);
        sysScroll.setPreferredSize(new Dimension(0, 60));
        gc.gridy = row++; gc.insets = new Insets(0, 0, 0, 0);
        gc.fill = GridBagConstraints.BOTH;
        p.add(sysScroll, gc);

        // Glue
        gc.gridy = row; gc.weighty = 1; gc.fill = GridBagConstraints.VERTICAL;
        p.add(Box.createGlue(), gc);

        return p;
    }

    private int addRow(JPanel p, GridBagConstraints gc, int row, String labelText, JComponent comp) {
        gc.gridwidth = 1; gc.weightx = 0;
        gc.fill = GridBagConstraints.NONE; gc.anchor = GridBagConstraints.WEST;
        JLabel lbl = new JLabel(labelText);
        lbl.setPreferredSize(new Dimension(85, 22));
        gc.gridx = 0; gc.gridy = row; gc.insets = new Insets(0, 0, 2, 6);
        p.add(lbl, gc);

        gc.gridx = 1; gc.weightx = 1; gc.fill = GridBagConstraints.HORIZONTAL;
        gc.insets = new Insets(0, 0, 2, 0);
        p.add(comp, gc);

        gc.gridx = 0; gc.gridwidth = 2; gc.weightx = 1;
        return row + 1;
    }

    private static JLabel sectionLabel(String text) {
        JLabel l = new JLabel(text);
        l.setFont(l.getFont().deriveFont(Font.BOLD));
        return l;
    }

    // -- Chat Area --

    private JPanel buildChatArea() {
        JPanel panel = new JPanel(new BorderLayout());

        chatPane = new JTextPane();
        chatPane.setEditable(false);
        chatPane.setBorder(BorderFactory.createEmptyBorder(10, 14, 10, 14));
        chatDoc = chatPane.getStyledDocument();
        initChatStyles();
        appendStyled("Load a model from the sidebar to start chatting.\n", "info");

        JScrollPane chatScroll = new JScrollPane(chatPane);
        chatScroll.getVerticalScrollBar().setUnitIncrement(16);
        panel.add(chatScroll, BorderLayout.CENTER);

        // Input
        JPanel inputBar = new JPanel(new BorderLayout(6, 0));
        inputBar.setBorder(BorderFactory.createEmptyBorder(6, 10, 6, 10));

        inputArea = new JTextArea(2, 1);
        inputArea.setLineWrap(true);
        inputArea.setWrapStyleWord(true);
        inputArea.setEnabled(false);
        inputArea.addKeyListener(new KeyAdapter() {
            @Override public void keyPressed(KeyEvent e) {
                if (e.getKeyCode() == KeyEvent.VK_ENTER && !e.isShiftDown()) {
                    e.consume();
                    onSendClick();
                }
            }
        });
        JScrollPane inputScroll = new JScrollPane(inputArea);
        inputScroll.setPreferredSize(new Dimension(0, 48));
        inputBar.add(inputScroll, BorderLayout.CENTER);

        sendBtn = new JButton("Send");
        sendBtn.setEnabled(false);
        sendBtn.setPreferredSize(new Dimension(80, 40));
        sendBtn.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                onSendClick();
            }
        });
        inputBar.add(sendBtn, BorderLayout.EAST);

        panel.add(inputBar, BorderLayout.SOUTH);
        return panel;
    }

    private void initChatStyles() {
        Style def = StyleContext.getDefaultStyleContext().getStyle(StyleContext.DEFAULT_STYLE);

        Style s = chatDoc.addStyle("user-label", def);
        StyleConstants.setBold(s, true);
        StyleConstants.setForeground(s, new Color(0x0055AA));
        StyleConstants.setFontSize(s, 12);

        s = chatDoc.addStyle("assistant-label", def);
        StyleConstants.setBold(s, true);
        StyleConstants.setForeground(s, new Color(0x227722));
        StyleConstants.setFontSize(s, 12);

        s = chatDoc.addStyle("user-text", def);
        StyleConstants.setFontFamily(s, "SansSerif");
        StyleConstants.setFontSize(s, 14);

        s = chatDoc.addStyle("assistant-text", def);
        StyleConstants.setFontFamily(s, "SansSerif");
        StyleConstants.setFontSize(s, 14);

        s = chatDoc.addStyle("info", def);
        StyleConstants.setForeground(s, Color.GRAY);
        StyleConstants.setItalic(s, true);
        StyleConstants.setFontSize(s, 12);

        s = chatDoc.addStyle("sep", def);
        StyleConstants.setFontSize(s, 6);
    }

    // -- Model management --

    private void loadModelList() {
        modelCombo.removeAllItems();
        Path dir = Paths.get(ggufDirectory);
        if (!Files.isDirectory(dir)) return;
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir, "*.gguf")) {
            List<ModelEntry> entries = new ArrayList<>();
            for (Path f : stream) {
                long sz = 0;
                try { sz = Files.size(f); } catch (IOException ignored) {}
                String arch = "unknown";
                try (GGUFFile gguf = GGUFParser.parse(f)) {
                    arch = gguf.getArchitecture();
                } catch (Exception ignored) {}
                boolean supported = SUPPORTED_ARCHS.contains(arch);
                entries.add(new ModelEntry(f.getFileName().toString(), f.toString(), sz, arch, supported));
            }
            Collections.sort(entries, new Comparator<ModelEntry>() {
                @Override
                public int compare(ModelEntry a, ModelEntry b) {
                    if (a.supported() != b.supported()) return a.supported() ? -1 : 1;
                    return a.name().compareToIgnoreCase(b.name());
                }
            });
            for (ModelEntry entry : entries) {
                modelCombo.addItem(entry);
            }
        } catch (IOException ignored) {}
    }

    private void loadModel() {
        if (generating) {
            JOptionPane.showMessageDialog(this,
                "Cannot load a new model while generation is in progress.\nPress Stop first.",
                "Generation Active", JOptionPane.WARNING_MESSAGE);
            return;
        }
        ModelEntry entry = (ModelEntry) modelCombo.getSelectedItem();
        if (entry == null) return;
        if (!entry.supported()) {
            JOptionPane.showMessageDialog(this,
                "Architecture '" + entry.arch() + "' is not yet supported.\nOnly Llama models can be loaded at this time.",
                "Unsupported Model", JOptionPane.WARNING_MESSAGE);
            return;
        }
        loadBtn.setEnabled(false);
        modelCombo.setEnabled(false);
        int ctxLen = ((Number) ctxSpinner.getValue()).intValue();
        Path modelPath = Paths.get(entry.path());

        // Build hardware plan and show confirmation
        final LLMEngine.HardwarePlan plan = LLMEngine.buildHardwarePlan(modelPath, ctxLen);
        String dialogTitle = plan.isRecommended() ? "Hardware Configuration" : "Warning: Not Recommended";
        int msgType = plan.isRecommended() ? JOptionPane.INFORMATION_MESSAGE : JOptionPane.WARNING_MESSAGE;
        String confirmMsg = plan.summary() + "\n\nProceed with loading?";

        int choice = JOptionPane.showConfirmDialog(this, confirmMsg, dialogTitle,
            JOptionPane.YES_NO_OPTION, msgType);
        if (choice != JOptionPane.YES_OPTION) {
            loadBtn.setEnabled(true);
            modelCombo.setEnabled(true);
            return;
        }

        // Build GPU config from plan or user overrides
        final GpuConfig gpuConfig;
        if (gpuCheckbox.isSelected() && !gpuDeviceList.isEmpty()) {
            // User has manually configured GPU settings - use those
            gpuConfig = new GpuConfig();
            gpuConfig.setEnabled(true);
            gpuConfig.setDeviceId(gpuDeviceCombo.getSelectedIndex());
            gpuConfig.setGpuLayers(((Number) gpuLayersSpinner.getValue()).intValue());
        } else if (plan.toGpuConfig() != null) {
            // Use the auto-configured plan
            gpuConfig = plan.toGpuConfig();
        } else {
            gpuConfig = new GpuConfig(); // CPU only
        }

        statusLabel.setText("Loading " + entry.name() + " (preloading weights)...");

        new SwingWorker<LLMEngine, Void>() {
            long elapsed;
            @Override protected LLMEngine doInBackground() throws Exception {
                long t0 = System.currentTimeMillis();
                LLMEngine eng = LLMEngine.load(modelPath, ctxLen, gpuConfig);
                elapsed = System.currentTimeMillis() - t0;
                return eng;
            }
            @Override protected void done() {
                try {
                    if (engine != null) engine.close();
                    engine = get();
                    ModelInfo info = engine.getModelInfo();
                    infoName.setText(info.name());
                    infoArch.setText(info.architecture());
                    infoLayers.setText(String.valueOf(info.blockCount()));
                    infoHeads.setText(info.headCount() + " / " + info.headCountKV());
                    infoEmbed.setText(String.valueOf(info.embeddingLength()));
                    infoCtx.setText(String.valueOf(info.contextLength()));
                    infoVocab.setText(String.format("%,d", info.vocabSize()));
                    infoLoad.setText(String.format("%.1fs", elapsed / 1000.0));
                    infoModelSize.setText(formatBytes(info.modelFileSize()));
                    infoKvCache.setText("~" + formatBytes(info.kvCacheEstimate()));
                    infoTotalRam.setText("~" + formatBytes(info.totalRamEstimate()));
                    // GPU info
                    String gpuName = engine.getGpuDeviceName();
                    int gpuLayers = engine.getGpuLayersUsed();
                    if (gpuName != null && gpuLayers > 0) {
                        infoGpuDevice.setText(gpuName);
                        String layersText = gpuLayers + " / " + info.blockCount();
                        if (engine.isMoeOptimizedGpu()) {
                            layersText += " (MoE: attn on GPU)";
                        }
                        infoGpuLayers.setText(layersText);
                    } else {
                        infoGpuDevice.setText("CPU only");
                        infoGpuLayers.setText("\u2014");
                    }
                    statusLabel.setText("Ready");
                    unloadBtn.setEnabled(true);
                    inputArea.setEnabled(true);
                    sendBtn.setEnabled(true);
                    chatDoc.remove(0, chatDoc.getLength());
                    appendStyled("Model loaded. Start chatting!\n", "info");
                    inputArea.requestFocusInWindow();
                } catch (Exception ex) {
                    Throwable cause = ex.getCause() != null ? ex.getCause() : ex;
                    cause.printStackTrace();
                    statusLabel.setText("Error loading model");
                    JOptionPane.showMessageDialog(LLMPlayerUI.this,
                        cause.getMessage(), "Model Load Error", JOptionPane.ERROR_MESSAGE);
                } finally {
                    loadBtn.setEnabled(true);
                    modelCombo.setEnabled(true);
                }
            }
        }.execute();
    }

    private void unloadModel() {
        if (engine != null) { engine.close(); engine = null; }
        statusLabel.setText("No model loaded");
        unloadBtn.setEnabled(false);
        inputArea.setEnabled(false);
        sendBtn.setEnabled(false);
        for (JLabel l : new JLabel[]{infoName, infoArch, infoLayers, infoHeads, infoEmbed, infoCtx, infoVocab, infoLoad,
                infoModelSize, infoKvCache, infoTotalRam, infoGpuDevice, infoGpuLayers})
            l.setText("\u2014");
    }

    // -- Chat / Generation --

    private void onSendClick() {
        if (generating) { stopRequested = true; return; }
        String text = inputArea.getText().trim();
        if (text.isEmpty() || engine == null) return;

        inputArea.setText("");
        inputArea.setEnabled(false);

        appendStyled("\n", "sep");
        appendStyled("USER\n", "user-label");
        appendStyled(text + "\n", "user-text");
        appendStyled("\n", "sep");
        appendStyled("ASSISTANT\n", "assistant-label");

        generating = true;
        stopRequested = false;
        statusLabel.setText("Processing prompt...");
        sendBtn.setText("Stop");
        loadBtn.setEnabled(false); // prevent model switch during generation

        float temp = ((Number) tempSpinner.getValue()).floatValue();
        int maxTok = ((Number) maxTokensSpinner.getValue()).intValue();
        int topK   = ((Number) topKSpinner.getValue()).intValue();
        float topP = ((Number) topPSpinner.getValue()).floatValue();
        float repP = ((Number) repPenaltySpinner.getValue()).floatValue();
        String sysMsg = systemMsgArea.getText().trim();

        SamplerConfig sc = new SamplerConfig(temp, topK, topP, repP, System.nanoTime());
        GenerationRequest req = GenerationRequest.builder()
            .prompt(text)
            .systemMessage(sysMsg.isEmpty() ? null : sysMsg)
            .maxTokens(maxTok)
            .samplerConfig(sc)
            .useChat(true)
            .build();

        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    System.out.println("[generate] Starting generation for: " + text);
                    SwingUtilities.invokeLater(new Runnable() {
                        @Override
                        public void run() {
                            appendStyled("(processing prompt...)\n", "info");
                        }
                    });
                    GenerationResponse resp = engine.generate(req, new StreamingCallback() {
                        @Override
                        public boolean onToken(String token, int id) {
                            if (stopRequested) return false;
                            SwingUtilities.invokeLater(new Runnable() {
                                @Override
                                public void run() {
                                    statusLabel.setText("Generating...");
                                    appendStyled(token, "assistant-text");
                                }
                            });
                            return true;
                        }
                    });
                    SwingUtilities.invokeLater(new Runnable() {
                        @Override
                        public void run() {
                            appendStyled(
                                String.format("\n[%d tokens, %.1f tok/s, %.1fs%s]\n",
                                    resp.tokenCount(), resp.tokensPerSecond(), resp.timeMs() / 1000.0,
                                    stopRequested ? ", stopped" : ""),
                                "info");
                        }
                    });
                } catch (final Throwable e) {
                    e.printStackTrace();
                    SwingUtilities.invokeLater(new Runnable() {
                        @Override
                        public void run() {
                            appendStyled("\n[Error: " + e.getMessage() + "]\n", "info");
                        }
                    });
                } finally {
                    SwingUtilities.invokeLater(new Runnable() {
                        @Override
                        public void run() {
                            generating = false;
                            statusLabel.setText("Ready");
                            sendBtn.setText("Send");
                            loadBtn.setEnabled(true);
                            inputArea.setEnabled(true);
                            inputArea.requestFocusInWindow();
                        }
                    });
                }
            }
        }, "llm-generate").start();
    }

    private void appendStyled(String text, String styleName) {
        try {
            chatDoc.insertString(chatDoc.getLength(), text, chatDoc.getStyle(styleName));
            chatPane.setCaretPosition(chatDoc.getLength());
        } catch (BadLocationException ignored) {}
    }

    // -- Web Server --

    private void startWebServer() {
        int port = ((Number) portSpinner.getValue()).intValue();
        try {
            webServer = new WebServer(port, ggufDirectory);
            webServer.start();
            webStartBtn.setEnabled(false);
            webStopBtn.setEnabled(true);
            portSpinner.setEnabled(false);
            webStatusLabel.setText("Running on port " + port);
            // Open browser
            try {
                Desktop.getDesktop().browse(java.net.URI.create("http://localhost:" + port));
            } catch (Exception ignored) {}
        } catch (IOException e) {
            webStatusLabel.setText("Error: " + e.getMessage());
        }
    }

    private void stopWebServer() {
        if (webServer != null) { webServer.stop(); webServer = null; }
        webStartBtn.setEnabled(true);
        webStopBtn.setEnabled(false);
        portSpinner.setEnabled(true);
        webStatusLabel.setText("Not running");
    }

    private void shutdown() {
        stopRequested = true;
        perfMonitor.stop();
        if (webServer != null) webServer.stop();
        if (engine != null) engine.close();
    }

    private static String formatBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        double mb = bytes / (1024.0 * 1024.0);
        if (mb < 1024) return String.format("%.0f MB", mb);
        return String.format("%.1f GB", mb / 1024.0);
    }

    /** JPanel that tells JScrollPane to always match viewport width. */
    static class ScrollablePanel extends JPanel implements Scrollable {
        ScrollablePanel(LayoutManager lm) { super(lm); }
        @Override public Dimension getPreferredScrollableViewportSize() { return getPreferredSize(); }
        @Override public int getScrollableUnitIncrement(Rectangle r, int o, int d) { return 16; }
        @Override public int getScrollableBlockIncrement(Rectangle r, int o, int d) { return 64; }
        @Override public boolean getScrollableTracksViewportWidth() { return true; }
        @Override public boolean getScrollableTracksViewportHeight() { return false; }
    }
}
