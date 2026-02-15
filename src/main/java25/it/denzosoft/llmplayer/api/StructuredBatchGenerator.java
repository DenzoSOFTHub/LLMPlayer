package it.denzosoft.llmplayer.api;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.StructuredTaskScope;
import java.util.concurrent.StructuredTaskScope.Subtask;

import it.denzosoft.llmplayer.evaluator.EvaluationResult;

/**
 * Java 25 batch generator using StructuredTaskScope.
 * Provides structured concurrency: automatic cancellation on error,
 * better observability, and proper lifecycle management.
 *
 * Loaded via reflection from LLMEngine when running on Java 25+.
 */
public final class StructuredBatchGenerator {

    private StructuredBatchGenerator() {}

    /**
     * Generate responses for multiple requests using StructuredTaskScope.
     * Each request runs in its own virtual thread with structured lifecycle.
     */
    public static List<GenerationResponse> generate(LLMEngine engine, List<GenerationRequest> requests) {
        try (var scope = StructuredTaskScope.open()) {
            List<Subtask<GenerationResponse>> subtasks = new ArrayList<>();
            for (GenerationRequest request : requests) {
                subtasks.add(scope.fork(() -> engine.generate(request)));
            }
            scope.join();
            List<GenerationResponse> responses = new ArrayList<>();
            for (Subtask<GenerationResponse> subtask : subtasks) {
                if (subtask.state() == Subtask.State.SUCCESS) {
                    responses.add(subtask.get());
                } else {
                    Throwable ex = subtask.exception();
                    responses.add(new GenerationResponse(
                        "Error: " + ex.getMessage(), 0, 0, 0, 0,
                        Collections.<EvaluationResult>emptyList()));
                }
            }
            return responses;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            List<GenerationResponse> errorResponses = new ArrayList<>();
            for (int i = 0; i < requests.size(); i++) {
                errorResponses.add(new GenerationResponse(
                    "Error: interrupted", 0, 0, 0, 0,
                    Collections.<EvaluationResult>emptyList()));
            }
            return errorResponses;
        }
    }
}
