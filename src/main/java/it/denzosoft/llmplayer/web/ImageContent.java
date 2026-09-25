package it.denzosoft.llmplayer.web;

import it.denzosoft.llmplayer.api.LLMEngine;

import java.util.Base64;
import java.util.List;
import java.util.Map;

/**
 * Image parts of chat messages for the OpenAI and Anthropic APIs. An image becomes an entry in the
 * request's image list plus an image marker ({@link LLMEngine#imageMarker()}) at its position in the
 * message text, which {@link LLMEngine} replaces with the vision embeddings.
 *
 * <p>Only inline images are accepted (OpenAI {@code data:} URLs, Anthropic {@code base64} sources):
 * the server does not fetch remote URLs.
 */
final class ImageContent {

    private ImageContent() {}

    /** Thrown for an image part that cannot be used; the message is safe to return to the client. */
    static final class BadImageException extends Exception {
        BadImageException(String message) { super(message); }
    }

    /**
     * Flatten an OpenAI message {@code content}: a string is returned as is; an array of parts has
     * its {@code text} parts joined and each {@code image_url} part replaced by an image marker, the
     * decoded bytes going to {@code images}.
     */
    @SuppressWarnings("unchecked")
    static String flattenOpenAI(Object content, List<byte[]> images) throws BadImageException {
        if (content == null) return null;
        if (content instanceof String) return (String) content;
        if (!(content instanceof List)) return content.toString();
        StringBuilder sb = new StringBuilder();
        for (Object partObj : (List<?>) content) {
            if (!(partObj instanceof Map)) continue;
            Map<String, Object> part = (Map<String, Object>) partObj;
            String type = (String) part.get("type");
            if ("text".equals(type)) {
                Object t = part.get("text");
                if (t != null) sb.append(t);
            } else if ("image_url".equals(type)) {
                Object iu = part.get("image_url");
                String url = iu instanceof Map ? (String) ((Map<String, Object>) iu).get("url")
                                               : (iu instanceof String ? (String) iu : null);
                images.add(decodeDataUrl(url));
                sb.append(LLMEngine.imageMarker());
            }
        }
        return sb.toString();
    }

    /** Anthropic {@code {"type":"image","source":{"type":"base64","data":...}}} block → bytes. */
    @SuppressWarnings("unchecked")
    static byte[] decodeAnthropicImage(Map<String, Object> block) throws BadImageException {
        Object src = block.get("source");
        if (!(src instanceof Map)) throw new BadImageException("image block without source");
        Map<String, Object> source = (Map<String, Object>) src;
        if (!"base64".equals(source.get("type"))) {
            throw new BadImageException("only base64 image sources are supported");
        }
        Object data = source.get("data");
        if (!(data instanceof String)) throw new BadImageException("image source without data");
        return decodeBase64((String) data);
    }

    private static byte[] decodeDataUrl(String url) throws BadImageException {
        if (url == null) throw new BadImageException("image_url without url");
        if (!url.startsWith("data:")) {
            throw new BadImageException("only data: image URLs are supported (base64-encode the image)");
        }
        int comma = url.indexOf(',');
        if (comma < 0 || !url.substring(0, comma).endsWith(";base64")) {
            throw new BadImageException("image data URL must be base64-encoded");
        }
        return decodeBase64(url.substring(comma + 1));
    }

    private static byte[] decodeBase64(String data) throws BadImageException {
        try {
            return Base64.getMimeDecoder().decode(data);
        } catch (IllegalArgumentException e) {
            throw new BadImageException("invalid base64 image data");
        }
    }
}
