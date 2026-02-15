package it.denzosoft.llmplayer.ui;

import javax.swing.*;
import java.awt.*;
import java.lang.management.ManagementFactory;

/**
 * Real-time CPU and heap memory monitor with a 60-second rolling graph.
 */
public class PerformanceMonitor extends JPanel {

    private static final int HISTORY = 60;
    private static final Color CPU_COLOR = new Color(0x4488DD);
    private static final Color MEM_COLOR = new Color(0xDD8844);
    private static final Color GRID_COLOR = new Color(0x3C3C3C);
    private static final Color BG_COLOR = new Color(0x1E1E1E);
    private static final Color TEXT_COLOR = new Color(0xCCCCCC);

    private final float[] cpuHistory = new float[HISTORY];
    private final float[] memHistory = new float[HISTORY]; // in MB
    private int writeIndex = 0;
    private float lastCpu = 0;
    private float lastMem = 0;
    private float maxMem = 256; // auto-scaled

    private final Timer timer;

    public PerformanceMonitor() {
        setPreferredSize(new Dimension(0, 110));
        setMinimumSize(new Dimension(100, 110));
        timer = new Timer(1000, e -> sample());
    }

    public void start() { timer.start(); }
    public void stop() { timer.stop(); }

    private void sample() {
        // CPU
        try {
            com.sun.management.OperatingSystemMXBean osBean =
                (com.sun.management.OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();
            double cpuLoad = osBean.getProcessCpuLoad();
            lastCpu = (float) (cpuLoad < 0 ? 0 : cpuLoad * 100);
        } catch (Exception e) {
            lastCpu = 0;
        }

        // Heap memory
        Runtime rt = Runtime.getRuntime();
        long usedBytes = rt.totalMemory() - rt.freeMemory();
        lastMem = usedBytes / (1024f * 1024f);

        cpuHistory[writeIndex] = lastCpu;
        memHistory[writeIndex] = lastMem;
        writeIndex = (writeIndex + 1) % HISTORY;

        // Auto-scale max memory
        float peak = 256;
        for (float m : memHistory) {
            if (m > peak) peak = m;
        }
        maxMem = Math.max(256, peak * 1.2f);

        repaint();
    }

    @Override
    protected void paintComponent(Graphics g0) {
        super.paintComponent(g0);
        Graphics2D g = (Graphics2D) g0;
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        int w = getWidth();
        int h = getHeight();
        int legendH = 18;
        int chartTop = 2;
        int chartBottom = h - legendH - 2;
        int chartH = chartBottom - chartTop;

        // Background
        g.setColor(BG_COLOR);
        g.fillRoundRect(0, 0, w, h, 6, 6);

        // Grid lines (25%, 50%, 75%)
        g.setColor(GRID_COLOR);
        g.setStroke(new BasicStroke(1));
        for (int pct = 25; pct <= 75; pct += 25) {
            int y = chartTop + chartH - (chartH * pct / 100);
            g.drawLine(4, y, w - 4, y);
        }

        // Draw lines
        if (writeIndex > 0 || cpuHistory[HISTORY - 1] > 0) {
            drawLine(g, cpuHistory, CPU_COLOR, chartTop, chartH, w, 100f);
            drawLine(g, memHistory, MEM_COLOR, chartTop, chartH, w, maxMem);
        }

        // Legend
        g.setFont(g.getFont().deriveFont(Font.PLAIN, 10f));
        int ly = h - 4;

        g.setColor(CPU_COLOR);
        g.fillOval(6, ly - 8, 8, 8);
        g.setColor(TEXT_COLOR);
        g.drawString(String.format("CPU %.0f%%", lastCpu), 18, ly);

        g.setColor(MEM_COLOR);
        g.fillOval(w / 2, ly - 8, 8, 8);
        g.setColor(TEXT_COLOR);
        g.drawString(String.format("Heap %.0f MB", lastMem), w / 2 + 12, ly);
    }

    private void drawLine(Graphics2D g, float[] data, Color color, int chartTop, int chartH, int w, float maxVal) {
        g.setColor(color);
        g.setStroke(new BasicStroke(1.5f));

        int margin = 4;
        float stepX = (w - 2f * margin) / (HISTORY - 1);

        int prevX = -1, prevY = -1;
        for (int i = 0; i < HISTORY; i++) {
            int dataIdx = (writeIndex + i) % HISTORY;
            float val = data[dataIdx];
            int x = margin + Math.round(i * stepX);
            int y = chartTop + chartH - Math.round(chartH * Math.min(val, maxVal) / maxVal);
            if (prevX >= 0) {
                g.drawLine(prevX, prevY, x, y);
            }
            prevX = x;
            prevY = y;
        }
    }
}
