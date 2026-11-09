(() => {
    const defaultColor = "#ff1493";
    const rgbColors = ["#ff0000", "#00ff00", "#0000ff"];

    function raster(mask, color, width) {
        const height = mask.length, columns = mask[0].length;
        const distance = new Int32Array(height * columns);
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < columns; x++) {
                const i = y * columns + x;
                if (mask[y][x] > 0.5) {
                    distance[i] = Math.min(x ? distance[i - 1] + 1 : 1,
                                           y ? distance[i - columns] + 1 : 1);
                }
            }
        }
        for (let y = height - 1; y >= 0; y--) {
            for (let x = columns - 1; x >= 0; x--) {
                const i = y * columns + x;
                if (distance[i]) {
                    distance[i] = Math.min(distance[i],
                        x + 1 < columns ? distance[i + 1] + 1 : 1,
                        y + 1 < height ? distance[i + columns] + 1 : 1);
                }
            }
        }
        const thickness = Math.ceil(width);
        return mask.map((row, y) => row.map((value, x) =>
            value > 0.5 && distance[y * columns + x] <= thickness ? [...color, 255] : [0, 0, 0, 0]));
    }

    function segments(values) {
        const code = values.reduce((bits, value, i) => bits | ((value > 0.5 ? 1 : 0) << i), 0);
        if (code === 5 || code === 10) {
            const determinant = (values[0] - 0.5) * (values[2] - 0.5) -
                                (values[1] - 0.5) * (values[3] - 0.5);
            return (determinant > 0 || (determinant === 0 && code === 5)) ? [[0, 1], [2, 3]] : [[3, 0], [1, 2]];
        }
        return [[], [[3, 0]], [[0, 1]], [[3, 1]], [[1, 2]], [], [[0, 2]], [[3, 2]],
                [[2, 3]], [[0, 2]], [], [[1, 2]], [[1, 3]], [[0, 1]], [[0, 3]], []][code];
    }

    function smooth(mask, color, width, scale) {
        const height = mask.length, columns = mask[0].length;
        const canvas = document.createElement("canvas");
        canvas.width = columns * scale;
        canvas.height = height * scale;
        const context = canvas.getContext("2d");
        context.scale(scale, scale);
        context.translate(0.5, 0.5);
        context.strokeStyle = `rgb(${color.join(",")})`;
        context.lineWidth = width;
        context.lineCap = "round";
        context.lineJoin = "round";
        const sample = (x, y) => {
            const value = mask[Math.max(0, Math.min(height - 1, y))][Math.max(0, Math.min(columns - 1, x))];
            return x < 0 || y < 0 || x >= columns || y >= height ? Math.min(value, 1 - value) : value;
        };
        context.beginPath();
        for (let y = -1; y < height; y++) {
            for (let x = -1; x < columns; x++) {
                const points = [[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]];
                const values = points.map(([px, py]) => sample(px, py));
                const crossing = edge => {
                    const next = (edge + 1) % 4;
                    const fraction = (0.5 - values[edge]) / (values[next] - values[edge]);
                    return points[edge].map((value, axis) => value + fraction * (points[next][axis] - value));
                };
                for (const [first, second] of segments(values)) {
                    context.moveTo(...crossing(first));
                    context.lineTo(...crossing(second));
                }
            }
        }
        context.stroke();
        const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
        return mask.map((row, y) => row.map((_, x) => {
            let alpha = 0;
            for (let dy = 0; dy < scale; dy++) {
                for (let dx = 0; dx < scale; dx++) {
                    alpha += pixels[((y * scale + dy) * canvas.width + x * scale + dx) * 4 + 3];
                }
            }
            alpha = Math.round(alpha / (scale * scale));
            return alpha ? [...color, alpha] : [0, 0, 0, 0];
        }));
    }

    function render(mask, hex, widthPercent, smoothness) {
        if (Array.isArray(mask[0][0])) {
            const colors = Array.isArray(hex) ? hex : mask.map(() => hex);
            const contours = mask.map((channel, i) => render(channel, colors[i], widthPercent, smoothness));
            return contours[0].map((row, y) => row.map((_, x) => {
                const pixels = contours.map(contour => contour[y][x]);
                const alpha = Math.max(...pixels.map(pixel => pixel[3]));
                return alpha ? [0, 1, 2].map(channel => Math.round(
                    Math.max(...pixels.map(pixel => pixel[channel] * pixel[3])) / alpha
                )).concat(alpha) : [0, 0, 0, 0];
            }));
        }
        if (Array.isArray(hex)) hex = hex[0];
        const color = hex.match(/[0-9a-f]{2}/gi).map(value => parseInt(value, 16));
        const width = Math.max(1, Math.min(mask.length, mask[0].length) * widthPercent / 100);
        return smoothness ? smooth(mask, color, width, Math.min(4, smoothness + 1)) : raster(mask, color, width);
    }

    function imageSource(pixels) {
        const canvas = document.createElement("canvas");
        canvas.width = pixels[0].length;
        canvas.height = pixels.length;
        const context = canvas.getContext("2d");
        const image = context.createImageData(canvas.width, canvas.height);
        let offset = 0;
        for (const row of pixels) {
            for (const pixel of row) {
                for (const channel of pixel) image.data[offset++] = channel;
            }
        }
        context.putImageData(image, 0, 0);
        return canvas.toDataURL("image/png");
    }

    function attach(plot) {
        if (plot._quacContoursAttached) return;
        plot._quacContoursAttached = true;
        const rgb = plot.data.some(trace => {
            const mask = trace.meta?.quac_contour?.mask;
            return mask?.length === 3 && Array.isArray(mask[0][0]);
        });
        const labels = rgb ? ["Red channel color", "Green channel color", "Blue channel color"] : ["Contour color"];
        const defaults = rgb ? rgbColors : [defaultColor];
        const controls = document.createElement("div");
        controls.className = "quac-contour-controls";
        controls.setAttribute("role", "region");
        controls.setAttribute("aria-label", "Contour settings");
        controls.innerHTML = `
            <style>
                .quac-contour-controls { display:flex; flex-wrap:wrap; align-items:center; gap:24px;
                    padding:12px 20px; margin:8px 0; width:fit-content; font:14px system-ui;
                    color:#24354b; background:white; border:1px solid #d7dce2; border-radius:6px; }
                .quac-contour-controls label { display:flex; align-items:center; gap:8px; }
                .quac-contour-controls input[type=number] { width:64px; }
                .quac-contour-controls input[type=color] { width:44px; height:28px; }
                .quac-contour-controls input[type=range] { width:110px; }
                .quac-contour-controls output { min-width:95px; }
            </style>
            ${labels.map((label, i) => `<label>${label}<input type="color" aria-label="${label}" value="${defaults[i]}"></label>`).join("")}
            <label>Contour width (%)<input type="number" aria-label="Contour width (%)" min="0" max="100"
                step="0.1" value="1" title="Percent of the smaller image dimension; at least one image pixel"></label>
            <label>Contour smoothness<input type="range" aria-label="Contour smoothness" min="0" max="3" step="1" value="0">
                <output>No smoothing</output></label>`;
        plot.before(controls);
        const colors = [...controls.querySelectorAll("[type=color]")];
        const width = controls.querySelector("[type=number]");
        const smoothness = controls.querySelector("[type=range]");
        const resolution = controls.querySelector("output");
        let busy = false, dirty = false;
        async function update() {
            if (busy) { dirty = true; return; }
            busy = true;
            try {
                do {
                    dirty = false;
                    const percent = Number.isFinite(width.valueAsNumber) ? Math.max(0, Math.min(100, width.valueAsNumber)) : 1;
                    const level = Number(smoothness.value);
                    resolution.value = level ? `${level + 1}× antialiasing` : "No smoothing";
                    const palette = colors.map(color => color.value);
                    const style = `${palette.join(",")}:${percent}:${level}`;
                    const smoothing = level ? "fast" : false;
                    const indices = [], images = [], metadata = [];
                    plot.data.forEach((trace, index) => {
                        const contour = trace.meta?.quac_contour;
                        if (!contour || (contour.style === style && trace.zsmooth === smoothing)) return;
                        indices.push(index);
                        images.push(imageSource(render(contour.mask, palette, percent, level)));
                        metadata.push({...trace.meta, quac_contour: {...contour, style}});
                    });
                    if (indices.length) await Plotly.restyle(plot, {source: images, meta: metadata, zsmooth: smoothing}, indices);
                } while (dirty);
            } finally { busy = false; }
        }
        const schedule = () => requestAnimationFrame(update);
        for (const input of [...colors, width, smoothness]) input.addEventListener("input", schedule);
        plot.on("plotly_afterplot", schedule);
        schedule();
    }

    globalThis.QuacContours = {render, attach};
})();
