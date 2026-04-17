package it.denzosoft.llmplayer.ui;

import it.denzosoft.llmplayer.api.LLMEngine;
import it.denzosoft.llmplayer.gguf.GGUFFile;
import it.denzosoft.llmplayer.gguf.GGUFParser;
import it.denzosoft.llmplayer.gpu.GpuConfig;
import it.denzosoft.llmplayer.model.ModelConfig;

import javax.swing.*;
import javax.swing.border.TitledBorder;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.*;
import java.nio.file.Path;

/**
 * Pre-load configuration dialog — shown when the user clicks "Load" on a model.
 *
 * Parses the GGUF header on the EDT (cheap — metadata only, no tensor read) to
 * produce a live summary of:
 *   - model architecture / dimensions / vocab / native context
 *   - projected RAM footprint (weights + activation buffers)
 *   - projected VRAM footprint for the chosen GPU layer count
 *   - projected KV cache size (FP32 vs Q8_0 toggle)
 *
 * Lets the user override the auto-detected plan:
 *   - context length (clamped to model's native max)
 *   - GPU layer count (−1 = auto, 0 = CPU only, N = force N layers)
 *   - MoE-optimized placement (attention on GPU, experts on CPU) — shown only for MoE models
 *   - Q8_0 quantized KV cache (sets -Dkv.q8=true before load)
 *
 * The dialog returns the chosen configuration via {@link #getContextLength()},
 * {@link #getGpuConfigOrNull()} and {@link #isKvQ8()}. The caller is responsible
 * for applying {@link System#setProperty} for {@code kv.q8} prior to engine load
 * because the KV mode is read at {@code InferenceState} construction time.
 *
 * This is a Phase-1 read-mostly dialog: the four settings cover the ~90 % of
 * model-loading decisions users make. Sampler tuning (temperature / top-k /
 * top-p / thinking / system prompt) remains in the main sidebar because those
 * values can be changed per-turn, not just at load time.
 */
public final class ModelLoadDialog extends JDialog {

    private final Path modelPath;
    private final LLMEngine.HardwarePlan basePlan;

    // Cached model facts (from GGUF quick-parse). Null if parse failed.
    private final ModelConfig config;
    private final long modelFileSize;
    private final int maxContextLength;
    private final int blockCount;
    private final boolean isMoE;

    // Controls
    private final JSpinner contextSpinner;
    private final JSpinner gpuLayersSpinner;
    private final JCheckBox moeOptimizedCheck;
    private final JCheckBox kvQ8Check;

    // Live-updated estimate labels
    private final JLabel vramEstimateLabel;
    private final JLabel ramEstimateLabel;
    private final JLabel kvCacheEstimateLabel;
    private final JLabel planSummaryLabel;

    // Result
    private boolean accepted;

    public ModelLoadDialog(JFrame parent, Path modelPath, LLMEngine.HardwarePlan plan,
                           int requestedContextLength) {
        super(parent, "Load model — configure", true);
        this.modelPath = modelPath;
        this.basePlan = plan;

        // Quick-parse GGUF header for model dimensions.
        ModelConfig cfg = null;
        long fileSize = 0;
        try {
            fileSize = java.nio.file.Files.size(modelPath);
            try (GGUFFile gguf = GGUFParser.parse(modelPath)) {
                cfg = ModelConfig.fromMetadata(gguf.getMetadata());
            }
        } catch (Exception ignored) {}
        this.config = cfg;
        this.modelFileSize = fileSize;
        this.maxContextLength = (cfg != null && cfg.contextLength() > 0) ? cfg.contextLength() : 4096;
        this.blockCount = (cfg != null) ? cfg.blockCount() : plan.totalLayers();
        this.isMoE = (cfg != null) && cfg.expertCount() > 0;

        int initialCtx = Math.min(Math.max(requestedContextLength, 512), maxContextLength);
        int initialGpuLayers = plan.gpuLayers();
        boolean initialMoe = plan.isMoeOptimized();
        boolean initialKvQ8 = "true".equals(System.getProperty("kv.q8"));

        // ---- Build UI ----
        setLayout(new BorderLayout(8, 8));
        ((JComponent) getContentPane()).setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        // Header
        JPanel header = new JPanel(new GridLayout(0, 1, 0, 2));
        String modelName = (cfg != null && cfg.name() != null) ? cfg.name() : modelPath.getFileName().toString();
        JLabel titleLabel = new JLabel(modelName);
        titleLabel.setFont(titleLabel.getFont().deriveFont(Font.BOLD, 14f));
        header.add(titleLabel);
        if (cfg != null) {
            header.add(new JLabel(
                cfg.architecture().name().toLowerCase() + " · "
                + blockCount + " layers · "
                + cfg.embeddingLength() + "d embed · "
                + cfg.headCount() + "/" + cfg.headCountKV() + " heads · "
                + String.format("%,d", cfg.vocabSize()) + " vocab · "
                + "ctx max " + String.format("%,d", maxContextLength)
                + (isMoE ? " · " + cfg.expertCount() + " experts" : "")));
        }
        header.add(new JLabel("File size: " + formatMB(modelFileSize)));
        add(header, BorderLayout.NORTH);

        // Centre: two columns — settings and live estimates
        JPanel centre = new JPanel(new GridLayout(1, 2, 10, 0));

        // Settings column
        JPanel settings = new JPanel(new GridBagLayout());
        settings.setBorder(new TitledBorder("Settings"));
        GridBagConstraints gc = new GridBagConstraints();
        gc.insets = new Insets(3, 4, 3, 4);
        gc.anchor = GridBagConstraints.WEST;
        gc.fill = GridBagConstraints.HORIZONTAL;
        int row = 0;

        gc.gridx = 0; gc.gridy = row; settings.add(new JLabel("Context length:"), gc);
        gc.gridx = 1; gc.weightx = 1; gc.gridy = row++;
        contextSpinner = new JSpinner(new SpinnerNumberModel(initialCtx, 512, maxContextLength, 512));
        settings.add(contextSpinner, gc);

        gc.gridx = 0; gc.weightx = 0; gc.gridy = row; settings.add(new JLabel("GPU layers (−1 = auto):"), gc);
        gc.gridx = 1; gc.weightx = 1; gc.gridy = row++;
        int gpuLayersMax = Math.max(blockCount, initialGpuLayers);
        gpuLayersSpinner = new JSpinner(new SpinnerNumberModel(initialGpuLayers, -1, gpuLayersMax, 1));
        settings.add(gpuLayersSpinner, gc);

        gc.gridx = 0; gc.weightx = 0; gc.gridy = row; settings.add(new JLabel("MoE-optimized:"), gc);
        gc.gridx = 1; gc.weightx = 1; gc.gridy = row++;
        moeOptimizedCheck = new JCheckBox("all attention on GPU, experts on CPU", initialMoe);
        moeOptimizedCheck.setEnabled(isMoE);
        if (!isMoE) {
            moeOptimizedCheck.setToolTipText("Only applies to MoE architectures (Qwen3MoE, DeepSeek2, GPT-OSS…).");
        }
        settings.add(moeOptimizedCheck, gc);

        gc.gridx = 0; gc.weightx = 0; gc.gridy = row; settings.add(new JLabel("KV cache Q8_0:"), gc);
        gc.gridx = 1; gc.weightx = 1; gc.gridy = row++;
        kvQ8Check = new JCheckBox("quantized KV cache (1.125 B/elem vs 4 B/elem)", initialKvQ8);
        kvQ8Check.setToolTipText("Reduces KV-cache VRAM ~3.56×. +28 % on DeepSeek2 MLA; 0-6 % CPU cost on dense models.");
        settings.add(kvQ8Check, gc);

        // Spacer
        gc.gridx = 0; gc.gridy = row++; gc.gridwidth = 2; gc.weighty = 1;
        settings.add(new JLabel(), gc);

        centre.add(settings);

        // Estimates column
        JPanel estimates = new JPanel(new GridBagLayout());
        estimates.setBorder(new TitledBorder("Projected footprint"));
        GridBagConstraints eg = new GridBagConstraints();
        eg.insets = new Insets(3, 4, 3, 4);
        eg.anchor = GridBagConstraints.WEST;
        eg.fill = GridBagConstraints.HORIZONTAL;
        int er = 0;

        eg.gridx = 0; eg.gridy = er;
        estimates.add(new JLabel("GPU device:"), eg);
        eg.gridx = 1; eg.weightx = 1; eg.gridy = er++;
        String gpuText = plan.isGpuAvailable() && plan.gpuDeviceName() != null
            ? plan.gpuDeviceName() + " (" + formatMB(plan.gpuVram()) + ")"
            : "not available";
        estimates.add(new JLabel(gpuText), eg);

        eg.gridx = 0; eg.weightx = 0; eg.gridy = er; estimates.add(new JLabel("Estimated VRAM:"), eg);
        eg.gridx = 1; eg.weightx = 1; eg.gridy = er++;
        vramEstimateLabel = new JLabel("—");
        estimates.add(vramEstimateLabel, eg);

        eg.gridx = 0; eg.weightx = 0; eg.gridy = er; estimates.add(new JLabel("Estimated RAM:"), eg);
        eg.gridx = 1; eg.weightx = 1; eg.gridy = er++;
        ramEstimateLabel = new JLabel("—");
        estimates.add(ramEstimateLabel, eg);

        eg.gridx = 0; eg.weightx = 0; eg.gridy = er; estimates.add(new JLabel("KV cache:"), eg);
        eg.gridx = 1; eg.weightx = 1; eg.gridy = er++;
        kvCacheEstimateLabel = new JLabel("—");
        estimates.add(kvCacheEstimateLabel, eg);

        eg.gridx = 0; eg.weightx = 0; eg.gridy = er; estimates.add(new JLabel("Plan:"), eg);
        eg.gridx = 1; eg.weightx = 1; eg.gridy = er++;
        planSummaryLabel = new JLabel("—");
        estimates.add(planSummaryLabel, eg);

        // Spacer
        eg.gridx = 0; eg.gridy = er++; eg.gridwidth = 2; eg.weighty = 1;
        estimates.add(new JLabel(), eg);

        centre.add(estimates);
        add(centre, BorderLayout.CENTER);

        // Bottom: buttons
        JPanel buttons = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        JButton resetBtn = new JButton("Reset to defaults");
        resetBtn.addActionListener(e -> {
            contextSpinner.setValue(Math.min(Math.max(requestedContextLength, 512), maxContextLength));
            gpuLayersSpinner.setValue(plan.gpuLayers());
            moeOptimizedCheck.setSelected(plan.isMoeOptimized());
            kvQ8Check.setSelected(false);
        });
        JButton loadBtn = new JButton("Load");
        loadBtn.addActionListener(e -> { accepted = true; setVisible(false); });
        JButton cancelBtn = new JButton("Cancel");
        cancelBtn.addActionListener(e -> { accepted = false; setVisible(false); });
        getRootPane().setDefaultButton(loadBtn);
        buttons.add(resetBtn);
        buttons.add(cancelBtn);
        buttons.add(loadBtn);
        add(buttons, BorderLayout.SOUTH);

        // Live-update estimates when any control changes
        ChangeListener updater = e -> updateEstimates();
        contextSpinner.addChangeListener(updater);
        gpuLayersSpinner.addChangeListener(updater);
        moeOptimizedCheck.addActionListener(e -> updateEstimates());
        kvQ8Check.addActionListener(e -> updateEstimates());
        updateEstimates();

        addWindowListener(new WindowAdapter() {
            @Override public void windowClosing(WindowEvent e) { accepted = false; }
        });

        pack();
        setLocationRelativeTo(parent);
    }

    /** Recompute the footprint labels from the current control values. */
    private void updateEstimates() {
        int ctx = getContextLength();
        int gpuLayers = getGpuLayers();
        boolean moe = moeOptimizedCheck.isSelected();
        boolean kvQ8 = kvQ8Check.isSelected();

        // Resolve "auto" (−1) for display purposes.
        int effectiveGpuLayers = gpuLayers;
        if (effectiveGpuLayers < 0) {
            effectiveGpuLayers = Math.min(basePlan.gpuLayers(), blockCount);
        }
        effectiveGpuLayers = Math.max(0, Math.min(effectiveGpuLayers, blockCount));

        long bytesPerLayer = (blockCount > 0) ? (modelFileSize / blockCount) : 0;
        long weightsVramBytes = (isMoE && moe)
            ? Math.min(modelFileSize, (long) (basePlan.gpuVram() * 0.80))
            : (long) effectiveGpuLayers * bytesPerLayer;
        long weightsRamBytes = Math.max(0, modelFileSize - weightsVramBytes);

        long kvBytes = estimateKvCacheBytes(ctx, kvQ8);
        long activationBytes = estimateActivationBytes();

        // KV cache goes to wherever the layer lives (on GPU for GPU layers, CPU for the rest).
        long kvOnGpu = (blockCount > 0) ? kvBytes * effectiveGpuLayers / blockCount : 0;
        long kvOnCpu = kvBytes - kvOnGpu;

        long totalVram = weightsVramBytes + kvOnGpu + activationBytes;
        long totalRam = weightsRamBytes + kvOnCpu + activationBytes;

        if (!basePlan.isGpuAvailable() || effectiveGpuLayers == 0) {
            vramEstimateLabel.setText("0 (CPU only)");
            ramEstimateLabel.setText(formatMB(modelFileSize + kvBytes + activationBytes));
        } else {
            vramEstimateLabel.setText(formatMB(totalVram)
                + (basePlan.gpuVram() > 0 ? " / " + formatMB(basePlan.gpuVram()) : "")
                + (totalVram > basePlan.gpuVram() * 0.95 ? "   \u26A0 tight" : ""));
            ramEstimateLabel.setText(formatMB(totalRam));
        }
        kvCacheEstimateLabel.setText(formatMB(kvBytes) + (kvQ8 ? " (Q8_0)" : " (FP32)"));

        StringBuilder planText = new StringBuilder();
        if (!basePlan.isGpuAvailable() || effectiveGpuLayers == 0) {
            planText.append("CPU only");
        } else if (isMoE && moe) {
            planText.append("MoE-optimized — attention on GPU, experts on CPU");
        } else if (effectiveGpuLayers >= blockCount) {
            planText.append("all ").append(blockCount).append(" layers on GPU");
        } else {
            planText.append(effectiveGpuLayers).append(" / ").append(blockCount)
                    .append(" layers on GPU (partial offload)");
        }
        planSummaryLabel.setText(planText.toString());
    }

    /** KV cache size (bytes) across all layers for the given context length + quant mode. */
    private long estimateKvCacheBytes(int contextLength, boolean kvQ8) {
        if (config == null) return 0;
        int kvDim = config.headCountKV() * (config.embeddingLength() / Math.max(config.headCount(), 1));
        // 2 = K and V. Q8_0: 1.125 B/elem (1 byte int8 + 2 bytes scale per 32-elem block → 36/32).
        double bytesPerElem = kvQ8 ? (36.0 / 32.0) : 4.0;
        return (long) (2.0 * blockCount * (long) contextLength * kvDim * bytesPerElem);
    }

    /** Rough activation / scratch buffer estimate — embedding + a few dim-sized buffers. */
    private long estimateActivationBytes() {
        if (config == null) return 0;
        // 8 × dim floats for gpuX / gpuXb / gpuXb2 / gpuQ / gpuK / gpuV / gpuHb / gpuHb2 equivalents,
        // plus the token-embedding table on CPU (vocab × dim × 2 bytes for F16 / F32 average).
        long scratch = 8L * config.embeddingLength() * Float.BYTES;
        // FFN scratch 2×ffnDim if we know it — use embeddingLength as conservative upper bound otherwise.
        scratch += 4L * config.embeddingLength() * Float.BYTES;
        return scratch;
    }

    private static String formatMB(long bytes) {
        if (bytes < 1024L * 1024) return String.format("%.1f KB", bytes / 1024.0);
        if (bytes < 1024L * 1024 * 1024) return String.format("%.0f MB", bytes / 1024.0 / 1024.0);
        return String.format("%.2f GB", bytes / 1024.0 / 1024.0 / 1024.0);
    }

    // ---- Results ----

    public boolean showAndWait() {
        setVisible(true);
        return accepted;
    }

    public int getContextLength() { return ((Number) contextSpinner.getValue()).intValue(); }
    public int getGpuLayers() { return ((Number) gpuLayersSpinner.getValue()).intValue(); }
    public boolean isMoeOptimizedSelected() { return moeOptimizedCheck.isSelected(); }
    public boolean isKvQ8() { return kvQ8Check.isSelected(); }

    /**
     * Build a GpuConfig from the dialog's current values, honouring the auto-plan when
     * the user leaves GPU layers on −1, or returning null when CPU-only was requested.
     */
    public GpuConfig getGpuConfigOrNull() {
        int layers = getGpuLayers();
        if (!basePlan.isGpuAvailable() || layers == 0) return null;
        GpuConfig cfg = new GpuConfig();
        cfg.setEnabled(true);
        cfg.setDeviceId(basePlan.gpuDeviceId());
        if (layers < 0) {
            cfg.setGpuLayers(basePlan.gpuLayers());
        } else {
            cfg.setGpuLayers(layers);
        }
        cfg.setMoeOptimized(isMoE && isMoeOptimizedSelected());
        return cfg;
    }

}
