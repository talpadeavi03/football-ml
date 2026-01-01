export default {
    async fetch(request, env) {
        const url = new URL(request.url);

        // Simple API Key Auth
        const apiKey = request.headers.get("x-api-key");
        if (apiKey !== env.API_KEY) {
            return new Response("Unauthorized", { status: 401 });
        }

        // Health check routing
        if (url.pathname === "/health") {
            return fetch(`${env.BACKEND_URL}/health`, request);
        }

        // Prediction routing with Canary support
        if (url.pathname === "/predict") {
            const rand = Math.random();
            const targetBackend = (env.CANARY_ENABLED === "true" && rand < parseFloat(env.CANARY_WEIGHT || "0"))
                ? env.CANARY_BACKEND_URL
                : env.BACKEND_URL;

            return fetch(`${targetBackend}/predict`, request);
        }

        return new Response("Not Found", { status: 404 });
    }
}
