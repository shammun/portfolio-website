"use client";

import { useEffect, useRef } from "react";

export default function ClimateAIBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let particlesArray: Particle[] = [];
    let animationId: number;

    // CONFIGURATION
    const PARTICLE_COUNT = 160;
    const CONNECTION_DISTANCE = 140;
    const MOUSE_RADIUS = 250;
    const SPEED_SCALE = 0.5;

    // Colors
    const COLOR_CLIMATE = "20, 184, 166"; // Teal
    const COLOR_AI = "99, 102, 241"; // Indigo

    const mouse = { x: null as number | null, y: null as number | null };
    let time = 0;

    // Resize Handling
    function resizeCanvas() {
      canvas!.width = window.innerWidth;
      canvas!.height = window.innerHeight;
    }

    // --- FLUID DYNAMICS ENGINE (Stream Function) ---
    function getAtmosphericVelocity(x: number, y: number, t: number) {
      const scale = 0.0012;
      const uvX = x * scale;
      const uvY = y * scale;
      const eps = 0.1;

      const psi1 = calculateStreamFunction(uvX, uvY - eps, t);
      const psi2 = calculateStreamFunction(uvX, uvY + eps, t);
      const psi3 = calculateStreamFunction(uvX - eps, uvY, t);
      const psi4 = calculateStreamFunction(uvX + eps, uvY, t);

      const u = -(psi2 - psi1) / (2 * eps);
      const v = (psi4 - psi3) / (2 * eps);

      return { vx: u, vy: v };
    }

    function calculateStreamFunction(x: number, y: number, t: number) {
      let psi = 0;
      // 1. ROSSBY WAVES
      psi += 2.5 * Math.sin(y * 3.0) * Math.cos(x * 2.2 + t * 0.15);
      // 2. OCEAN GYRES
      psi += 2.0 * Math.sin(x * 1.2 + t * 0.05) * Math.sin(y * 1.5);
      // 3. EDDIES
      psi += 0.6 * Math.sin(x * 4.0 - t * 0.3) * Math.cos(y * 3.5 + t * 0.2);
      // 4. ZONAL FLOW
      psi -= 0.4 * y;
      return psi;
    }

    // Particle Class
    class Particle {
      x: number = 0;
      y: number = 0;
      type: "CLIMATE" | "AI" = "CLIMATE";
      size: number = 0;
      life: number = 0;
      maxLife: number = 0;
      vx: number = 0;
      vy: number = 0;
      alpha: number = 0;

      constructor() {
        this.init(true);
      }

      init(randomScreenPos = false) {
        if (randomScreenPos) {
          this.x = Math.random() * canvas!.width;
          this.y = Math.random() * canvas!.height;
        } else {
          this.x = Math.random() * canvas!.width;
          this.y = Math.random() * canvas!.height;
        }

        this.type = Math.random() > 0.5 ? "CLIMATE" : "AI";
        this.size =
          this.type === "CLIMATE"
            ? Math.random() * 2.5 + 2.0
            : Math.random() * 2.0 + 1.5;

        this.life = 0;
        this.maxLife = Math.random() * 500 + 300;
        this.vx = 0;
        this.vy = 0;
        this.alpha = 0;
      }

      update() {
        this.life++;

        if (this.life < 80) this.alpha += 0.01;
        if (this.life > this.maxLife - 80) this.alpha -= 0.01;
        if (this.alpha < 0) this.alpha = 0;
        if (this.alpha > 0.9) this.alpha = 0.9;

        if (this.life > this.maxLife) {
          this.init(false);
          return;
        }

        const velocity = getAtmosphericVelocity(this.x, this.y, time);
        const inertia = this.type === "CLIMATE" ? 0.03 : 0.02;

        this.vx = this.vx * (1 - inertia) + velocity.vx * SPEED_SCALE * inertia;
        this.vy = this.vy * (1 - inertia) + velocity.vy * SPEED_SCALE * inertia;

        this.x += this.vx;
        this.y += this.vy;

        // Mouse interaction
        if (mouse.x != null && mouse.y != null) {
          const dx = this.x - mouse.x;
          const dy = this.y - mouse.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < MOUSE_RADIUS) {
            const force = (MOUSE_RADIUS - dist) / MOUSE_RADIUS;
            this.x += (dy / dist) * force * 1.5;
            this.y -= (dx / dist) * force * 1.5;
          }
        }

        // Wrapping
        if (this.x > canvas!.width + 20) this.x = -20;
        if (this.x < -20) this.x = canvas!.width + 20;
        if (this.y > canvas!.height + 20) this.y = -20;
        if (this.y < -20) this.y = canvas!.height + 20;
      }

      draw() {
        if (!ctx) return;
        // Outer Glow
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size * 2, 0, Math.PI * 2);
        const color = this.type === "CLIMATE" ? COLOR_CLIMATE : COLOR_AI;
        ctx.fillStyle = `rgba(${color}, ${this.alpha * 0.15})`;
        ctx.fill();

        // Solid Core
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${color}, ${this.alpha})`;
        ctx.fill();

        // White center
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size * 0.3, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 255, 255, ${this.alpha * 0.8})`;
        ctx.fill();
      }
    }

    function init() {
      particlesArray = [];
      const count = Math.min(
        PARTICLE_COUNT,
        (window.innerWidth * window.innerHeight) / 9000
      );
      for (let i = 0; i < count; i++) {
        particlesArray.push(new Particle());
      }
    }

    function animate() {
      if (!ctx || !canvas) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      time += 0.001;

      for (let i = 0; i < particlesArray.length; i++) {
        particlesArray[i].update();
        particlesArray[i].draw();

        // Connections
        for (let j = i + 1; j < particlesArray.length; j++) {
          const dx = particlesArray[i].x - particlesArray[j].x;
          const dy = particlesArray[i].y - particlesArray[j].y;
          const distSq = dx * dx + dy * dy;
          const connDistSq = CONNECTION_DISTANCE * CONNECTION_DISTANCE;

          if (distSq < connDistSq) {
            const dist = Math.sqrt(distSq);
            let opacity = 1 - dist / CONNECTION_DISTANCE;
            opacity = opacity * opacity;
            opacity *= Math.min(
              particlesArray[i].alpha,
              particlesArray[j].alpha
            );

            if (opacity > 0.05) {
              ctx.beginPath();
              const gradient = ctx.createLinearGradient(
                particlesArray[i].x,
                particlesArray[i].y,
                particlesArray[j].x,
                particlesArray[j].y
              );

              const c1 =
                particlesArray[i].type === "CLIMATE" ? COLOR_CLIMATE : COLOR_AI;
              const c2 =
                particlesArray[j].type === "CLIMATE" ? COLOR_CLIMATE : COLOR_AI;

              gradient.addColorStop(0, `rgba(${c1}, ${opacity * 0.4})`);
              gradient.addColorStop(1, `rgba(${c2}, ${opacity * 0.4})`);

              ctx.strokeStyle = gradient;
              ctx.lineWidth = 1.2;
              ctx.moveTo(particlesArray[i].x, particlesArray[i].y);
              ctx.lineTo(particlesArray[j].x, particlesArray[j].y);
              ctx.stroke();
            }
          }
        }
      }
      animationId = requestAnimationFrame(animate);
    }

    // Event handlers
    const handleResize = () => {
      resizeCanvas();
      init();
    };

    const handleMouseMove = (e: MouseEvent) => {
      mouse.x = e.x;
      mouse.y = e.y;
    };

    const handleMouseOut = () => {
      mouse.x = null;
      mouse.y = null;
    };

    window.addEventListener("resize", handleResize);
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseout", handleMouseOut);

    // Start
    resizeCanvas();
    init();
    animate();

    // Cleanup
    return () => {
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseout", handleMouseOut);
      cancelAnimationFrame(animationId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full block"
    />
  );
}
