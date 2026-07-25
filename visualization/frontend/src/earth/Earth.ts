/**
 * Earth rendering with real NASA texture and atmospheric effects
 * Uses high-quality texture - place earth_daymap.jpg in public/textures/
 */

import * as THREE from 'three';

// Local Earth texture path (place the image in public/textures/)
const EARTH_TEXTURE_URL = '/textures/earth_daymap.jpg';

// Atmosphere shader for realistic glow
const atmosphereVertexShader = `
    varying vec3 vNormal;
    varying vec3 vPosition;
    
    void main() {
        vNormal = normalize(normalMatrix * normal);
        vPosition = (modelViewMatrix * vec4(position, 1.0)).xyz;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
`;

const atmosphereFragmentShader = `
    uniform vec3 lightDirection;
    varying vec3 vNormal;
    varying vec3 vPosition;
    
    void main() {
        vec3 viewDir = normalize(-vPosition);
        float fresnel = pow(1.0 - max(dot(viewDir, vNormal), 0.0), 3.5);
        
        vec3 atmosphereColor = vec3(0.4, 0.7, 1.0);
        vec3 sunsetColor = vec3(1.0, 0.5, 0.3);
        
        float sunAngle = dot(vNormal, lightDirection) * 0.5 + 0.5;
        vec3 finalColor = mix(atmosphereColor, sunsetColor, pow(1.0 - sunAngle, 2.0) * 0.4);
        
        float alpha = fresnel * 0.7;
        
        gl_FragColor = vec4(finalColor, alpha);
    }
`;

// Cloud shader
const cloudVertexShader = `
    varying vec2 vUv;
    varying vec3 vNormal;
    
    void main() {
        vUv = uv;
        vNormal = normalize(normalMatrix * normal);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
`;

const cloudFragmentShader = `
    uniform float time;
    varying vec2 vUv;
    varying vec3 vNormal;
    
    float hash(vec2 p) {
        return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
    }
    
    float noise(vec2 p) {
        vec2 i = floor(p);
        vec2 f = fract(p);
        f = f * f * (3.0 - 2.0 * f);
        
        float a = hash(i);
        float b = hash(i + vec2(1.0, 0.0));
        float c = hash(i + vec2(0.0, 1.0));
        float d = hash(i + vec2(1.0, 1.0));
        
        return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
    }
    
    float fbm(vec2 p) {
        float value = 0.0;
        float amplitude = 0.5;
        for (int i = 0; i < 5; i++) {
            value += amplitude * noise(p);
            p *= 2.0;
            amplitude *= 0.5;
        }
        return value;
    }
    
    void main() {
        vec2 uv = vUv * 6.0;
        uv.x += time * 0.015;
        
        float clouds = fbm(uv);
        float detail = fbm(uv * 2.0 - time * 0.008);
        clouds = clouds * 0.7 + detail * 0.3;
        clouds = smoothstep(0.35, 0.65, clouds);
        
        // More clouds near equator and ITCZ
        float lat = abs(vUv.y - 0.5) * 2.0;
        float tropicalBoost = 1.0 - smoothstep(0.0, 0.3, lat);
        clouds *= 0.7 + tropicalBoost * 0.5;
        
        vec3 lightDir = normalize(vec3(1.0, 0.3, 0.8));
        float lighting = dot(vNormal, lightDir) * 0.3 + 0.7;
        
        vec3 cloudColor = vec3(1.0) * lighting;
        float alpha = clouds * 0.45;
        
        gl_FragColor = vec4(cloudColor, alpha);
    }
`;

// Outer glow shader
const glowVertexShader = `
    varying vec3 vNormal;
    
    void main() {
        vNormal = normalize(normalMatrix * normal);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
`;

const glowFragmentShader = `
    varying vec3 vNormal;
    
    void main() {
        float intensity = pow(0.65 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 4.0);
        vec3 glowColor = vec3(0.3, 0.6, 1.0);
        gl_FragColor = vec4(glowColor * intensity, intensity * 0.6);
    }
`;

export function createEarth(): THREE.Group {
    const earthGroup = new THREE.Group();
    
    // Load the real Earth texture
    const textureLoader = new THREE.TextureLoader();
    
    // Create Earth mesh with real texture
    const earthGeometry = new THREE.SphereGeometry(1, 128, 128);
    
    // Load the Earth daymap texture
    const earthTexture = textureLoader.load(
        EARTH_TEXTURE_URL,
        (texture) => {
            console.log('✓ Earth texture loaded successfully');
            texture.anisotropy = 16;
        },
        undefined,
        (error) => {
            console.error('Failed to load Earth texture:', error);
        }
    );
    
    // Configure texture for better quality
    earthTexture.colorSpace = THREE.SRGBColorSpace;
    
    // Create Earth material with the real texture
    const earthMaterial = new THREE.MeshStandardMaterial({
        map: earthTexture,
        roughness: 0.8,
        metalness: 0.1,
    });
    
    const earthMesh = new THREE.Mesh(earthGeometry, earthMaterial);
    earthMesh.name = 'earth';
    earthMesh.receiveShadow = true;
    earthGroup.add(earthMesh);
    
    // Inner atmosphere layer
    const atmosphereGeometry = new THREE.SphereGeometry(1.012, 64, 64);
    const atmosphereMaterial = new THREE.ShaderMaterial({
        vertexShader: atmosphereVertexShader,
        fragmentShader: atmosphereFragmentShader,
        uniforms: {
            lightDirection: { value: new THREE.Vector3(1, 0.3, 0.8).normalize() },
        },
        transparent: true,
        blending: THREE.AdditiveBlending,
        side: THREE.FrontSide,
        depthWrite: false,
    });
    const atmosphere = new THREE.Mesh(atmosphereGeometry, atmosphereMaterial);
    atmosphere.name = 'atmosphere';
    earthGroup.add(atmosphere);
    
    // Cloud layer
    const cloudGeometry = new THREE.SphereGeometry(1.018, 64, 64);
    const cloudMaterial = new THREE.ShaderMaterial({
        vertexShader: cloudVertexShader,
        fragmentShader: cloudFragmentShader,
        uniforms: {
            time: { value: 0.0 },
        },
        transparent: true,
        depthWrite: false,
    });
    const clouds = new THREE.Mesh(cloudGeometry, cloudMaterial);
    clouds.name = 'clouds';
    earthGroup.add(clouds);
    
    // Outer atmospheric glow
    const glowGeometry = new THREE.SphereGeometry(1.15, 48, 48);
    const glowMaterial = new THREE.ShaderMaterial({
        vertexShader: glowVertexShader,
        fragmentShader: glowFragmentShader,
        blending: THREE.AdditiveBlending,
        side: THREE.BackSide,
        transparent: true,
        depthWrite: false,
    });
    const glow = new THREE.Mesh(glowGeometry, glowMaterial);
    glow.name = 'glow';
    earthGroup.add(glow);
    
    // Second outer glow for more dramatic effect
    const outerGlowGeometry = new THREE.SphereGeometry(1.25, 32, 32);
    const outerGlowMaterial = new THREE.ShaderMaterial({
        vertexShader: glowVertexShader,
        fragmentShader: `
            varying vec3 vNormal;
            void main() {
                float intensity = pow(0.5 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 5.0);
                vec3 glowColor = vec3(0.2, 0.5, 0.95);
                gl_FragColor = vec4(glowColor * intensity, intensity * 0.35);
            }
        `,
        blending: THREE.AdditiveBlending,
        side: THREE.BackSide,
        transparent: true,
        depthWrite: false,
    });
    const outerGlow = new THREE.Mesh(outerGlowGeometry, outerGlowMaterial);
    outerGlow.name = 'outerGlow';
    earthGroup.add(outerGlow);
    
    return earthGroup;
}

export function updateEarth(earth: THREE.Group, delta: number): void {
    const time = Date.now() * 0.001;
    
    // Rotate Earth slowly
    earth.rotation.y += delta * 0.012;
    
    // Rotate clouds slightly faster than Earth
    const clouds = earth.getObjectByName('clouds') as THREE.Mesh;
    if (clouds) {
        clouds.rotation.y += delta * 0.016;
        if (clouds.material instanceof THREE.ShaderMaterial) {
            clouds.material.uniforms.time.value = time;
        }
    }
}
