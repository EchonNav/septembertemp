//  EchoNav_Urban_Analyzer.swift
//  testing_september
//
//  Drop-in replacement that augments your current app with:
//  1) Urban scene analysis from ARMeshAnchor (LiDAR mesh + classifications)
//  2) Camera object detection via Vision/CoreML (e.g., YOLO) with 3D localization via raycasts
//  3) Natural-language descriptions + Text-to-Speech ("Poteau à 2 m, légèrement à gauche")
//  4) Lightweight rate limiting and HUD wiring
//
//  Notes:
//  - Plug your own .mlmodel (e.g., YOLOv5/v8 converted to CoreML). See VisionObjectDetector.model below.
//  - Requires: iOS 16+ for best mesh classifications; works on LiDAR devices.
//  - Keep your existing proximity audio; this file only adds analysis + narration.
//

import SwiftUI
import RealityKit
import ARKit
import AVFoundation
import Combine
import Vision
import CoreML

@main
struct testing_septemberApp: App {
    var body: some SwiftUI.Scene {
        WindowGroup { ProximityRootView() }
    }
}

// MARK: - Root View
struct ProximityRootView: View {
    @StateObject private var proximityVM = ProximityViewModel()
    var body: some View {
        ZStack(alignment: .top) {
            ARViewContainer(viewModel: proximityVM).ignoresSafeArea()
            HUD(viewModel: proximityVM).padding(12)
        }
        .onAppear { proximityVM.start() }
        .onDisappear { proximityVM.stop() }
    }
}

// MARK: - ViewModel
final class ProximityViewModel: NSObject, ObservableObject {
    // HUD
    @Published var distanceText: String = "—"
    @Published var statusText: String = "Initialisation…"
    @Published var isLiDARAvailable: Bool = false
    @Published var warningLevel: WarningLevel = .none
    @Published var lastDescription: String = "—"

    // Tuning
    let nearThreshold: Float = 0.6
    let midThreshold: Float  = 1.2
    let maxSense: Float      = 3.0

    fileprivate weak var arView: ARView?
    fileprivate var audio: ProximityAudio = ProximityAudio()
    fileprivate var haptics = UINotificationFeedbackGenerator()
    fileprivate var lastCrossedNear = false

    // Analysis components
    fileprivate let analyzer = UrbanAnalyzer()
    fileprivate let narrator = DescriptionSpeaker()
    fileprivate let detector = VisionObjectDetector()

    // Rate limiting for narration
    fileprivate var lastNarrationTime: CFTimeInterval = 0
    fileprivate let narrationCooldown: CFTimeInterval = 2.0 // seconds

    func start() {
        guard ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) else {
            statusText = "Appareil incompatible (pas de LiDAR)."
            isLiDARAvailable = false
            return
        }
        isLiDARAvailable = true
        statusText = "Recherche d’obstacles…"
        haptics.prepare()
        audio.startEngine()
    }

    func stop() {
        audio.stopEngine()
        arView?.session.pause()
    }
}

// MARK: - AR Container
struct ARViewContainer: UIViewRepresentable {
    let viewModel: ProximityViewModel

    func makeUIView(context: Context) -> ARView {
        let view = ARView(frame: .zero)
        viewModel.arView = view
        view.session.delegate = context.coordinator

        let config = ARWorldTrackingConfiguration()
        config.sceneReconstruction = .meshWithClassification
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
        }
        config.environmentTexturing = .automatic
        view.debugOptions = [] // [.showSceneUnderstanding]
        view.session.run(config, options: [.resetTracking, .removeExistingAnchors])
        return view
    }

    func updateUIView(_ uiView: ARView, context: Context) {}
    func makeCoordinator() -> Coordinator { Coordinator(viewModel: viewModel) }

    // MARK: - Coordinator
    final class Coordinator: NSObject, ARSessionDelegate {
        private let vm: ProximityViewModel
        private var lastSampleTime: CFTimeInterval = 0
        private let sampleHz: Double = 10 // keep proximity at ~10 Hz
        private var lastAnalysisTime: CFTimeInterval = 0
        private let analysisHz: Double = 1 // heavier analysis less often

        init(viewModel: ProximityViewModel) { self.vm = viewModel }

        func session(_ session: ARSession, didUpdate frame: ARFrame) {
            let now = frame.timestamp
            guard let arView = vm.arView else { return }

            // 1) Proximity via central raycast (existing behavior)
            if now - lastSampleTime >= 1.0 / sampleHz {
                lastSampleTime = now
                let center = CGPoint(x: arView.bounds.midX, y: arView.bounds.midY)
                let results = arView.raycast(from: center, allowing: .estimatedPlane, alignment: .any)
                if let result = results.first {
                    let camT = frame.camera.transform.columns.3
                    let resT = result.worldTransform.columns.3
                    let camPos = SIMD3<Float>(camT.x, camT.y, camT.z)
                    let hitPos = SIMD3<Float>(resT.x, resT.y, resT.z)
                    let d = simd_distance(camPos, hitPos)
                    handleDistance(d)
                } else { handleNoHit() }
            }

            // 2) Urban analysis (mesh + detections)
            if now - lastAnalysisTime >= 1.0 / analysisHz {
                lastAnalysisTime = now

                // Mesh-based cues (walls/doors/windows/floor etc.)
                let meshHints = vm.analyzer.analyzeMeshes(session: session, camera: frame.camera)

                // Vision object detection (if model available)
                let visionHints = vm.detector.detectObjects(on: frame, in: arView)

                // Merge & choose the best description (closest, salient, not spammy)
                let merged = (meshHints + visionHints).sorted { $0.distanceMeters < $1.distanceMeters }

                if let best = merged.first { deliver(best, at: now) }
            }
        }

        private func deliver(_ hint: UrbanHint, at now: CFTimeInterval) {
            vm.lastDescription = hint.localized
            if now - vm.lastNarrationTime > vm.narrationCooldown {
                vm.narrator.speak(hint.localized)
                vm.lastNarrationTime = now
            }
        }

        // Proximity mapping
        private func handleDistance(_ d: Float) {
            let clamped = max(0, min(d, vm.maxSense))
            vm.distanceText = String(format: "%.2f m", clamped)

            if d < vm.nearThreshold {
                vm.statusText = "DANGER: obstacle très proche"
                vm.warningLevel = .high
                vm.audio.setMode(.continuous(frequency: mapFreq(for: d)))
                if !vm.lastCrossedNear { vm.haptics.notificationOccurred(.warning); vm.lastCrossedNear = true }
            } else if d < vm.midThreshold {
                vm.statusText = "Attention: obstacle proche"
                vm.warningLevel = .medium
                vm.audio.setMode(.beep(interval: mapInterval(for: d), frequency: mapFreq(for: d)))
                if vm.lastCrossedNear { vm.lastCrossedNear = false }
            } else if d <= vm.maxSense {
                vm.statusText = "Zone dégagée"
                vm.warningLevel = .low
                vm.audio.setMode(.beep(interval: 1.0, frequency: 600))
                vm.lastCrossedNear = false
            } else { handleNoHit() }
        }

        private func handleNoHit() {
            vm.distanceText = "—"
            vm.statusText = "Aucun obstacle à portée"
            vm.warningLevel = .none
            vm.audio.setMode(.silent)
            vm.lastCrossedNear = false
        }

        private func mapInterval(for d: Float) -> Double {
            let dMin: Float = 0.2
            let dMax: Float = max(vm.midThreshold, dMin + 0.01)
            let tMin: Double = 0.15
            let tMax: Double = 1.2
            let t = Double((d - dMin) / (dMax - dMin))
            return max(tMin, min(tMax, t * (tMax - tMin) + tMin))
        }

        private func mapFreq(for d: Float) -> Double {
            let dMin: Float = 0.2
            let dMax: Float = max(vm.maxSense, dMin + 0.01)
            let fNear: Double = 1200
            let fFar: Double = 300
            let t = Double((min(max(d, dMin), dMax) - dMin) / (dMax - dMin))
            return fNear + (fFar - fNear) * t
        }
    }
}

// MARK: - HUD
struct HUD: View {
    @ObservedObject var viewModel: ProximityViewModel
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("EchoNav", systemImage: "wave.3.right.circle")
                .font(.title2.weight(.semibold))
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(.ultraThinMaterial, in: Capsule())

            HStack(spacing: 12) {
                Text(viewModel.distanceText)
                    .font(.system(size: 36, weight: .bold, design: .rounded))
                    .monospacedDigit()
                StatusBadge(level: viewModel.warningLevel, text: viewModel.statusText)
            }
            .padding(10)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))

            // Live textual description from analyzer
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: "speaker.wave.2.fill")
                Text(viewModel.lastDescription)
                    .font(.subheadline)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(10)
            .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        }
        .foregroundStyle(.primary)
    }
}

enum WarningLevel { case none, low, medium, high }

struct StatusBadge: View {
    let level: WarningLevel
    let text: String
    private var color: Color {
        switch level { case .none: return .gray; case .low: return .green; case .medium: return .orange; case .high: return .red }
    }
    var body: some View {
        HStack(spacing: 6) {
            Circle().frame(width: 10, height: 10).foregroundStyle(color)
            Text(text).font(.subheadline.weight(.medium))
        }
        .padding(.horizontal, 10).padding(.vertical, 6)
        .background(.thinMaterial, in: Capsule())
    }
}

// MARK: - Audio (unchanged from your version)
final class ProximityAudio {
    enum Mode { case silent, beep(interval: Double, frequency: Double), continuous(frequency: Double) }
    private let engine = AVAudioEngine()
    private var source: AVAudioSourceNode?
    private var timer: DispatchSourceTimer?
    private let sampleRate: Double = 44_100
    private var currentMode: Mode = .silent
    private var phase: Double = 0
    private var isBeeping = false
    private let beepDuration: Double = 0.08

    func startEngine() {
        guard engine.isRunning == false else { return }
        let src = AVAudioSourceNode { [weak self] _, _, frameCount, audioBufferList -> OSStatus in
            guard let self else { return noErr }
            guard let abl = UnsafeMutableAudioBufferListPointer(audioBufferList).first else { return noErr }
            let bufferPointer = abl.mData!.assumingMemoryBound(to: Float.self)
            let frames = Int(frameCount)
            var freq: Double = 0
            var renderTone = false
            switch self.currentMode {
            case .silent: renderTone = false
            case .continuous(let f): freq = f; renderTone = true
            case .beep(_, let f): freq = f; renderTone = self.isBeeping
            }
            if renderTone && freq > 0 {
                let twoPi = 2 * Double.pi
                for n in 0..<frames {
                    let sample = Float(sin(self.phase)) * 0.25
                    bufferPointer[n] = sample
                    self.phase += twoPi * freq / self.sampleRate
                    if self.phase > twoPi { self.phase -= twoPi }
                }
            } else {
                for n in 0..<frames { bufferPointer[n] = 0 }
            }
            return noErr
        }
        self.source = src
        let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
        engine.attach(src)
        engine.connect(src, to: engine.mainMixerNode, format: format)
        do { try engine.start() } catch { print("Audio engine error: \(error)") }
    }

    func stopEngine() { timer?.cancel(); timer = nil; engine.stop() }

    func setMode(_ mode: Mode) {
        if case .beep(let interval, let f) = mode { startBeepTimer(interval: interval, frequency: f) } else { stopBeepTimer() }
        currentMode = mode
    }

    private func startBeepTimer(interval: Double, frequency: Double) {
        stopBeepTimer()
        currentMode = .beep(interval: interval, frequency: frequency)
        let t = DispatchSource.makeTimerSource()
        t.schedule(deadline: .now(), repeating: interval)
        t.setEventHandler { [weak self] in
            guard let self else { return }
            self.isBeeping = true
            DispatchQueue.global().asyncAfter(deadline: .now() + self.beepDuration) { self.isBeeping = false }
        }
        t.resume(); timer = t
    }

    private func stopBeepTimer() { timer?.cancel(); timer = nil; isBeeping = false }
}

// MARK: - Urban Analysis Models
struct UrbanHint {
    let label: String
    let distanceMeters: Float
    let bearingDeg: Float

    var localized: String {
        let dir = Self.directionWord(for: bearingDeg)
        let d = String(format: "%.1f", distanceMeters)
        return "\(label.capitalized) détecté à \(d) m, \(dir)."
    }

    static func directionWord(for bearing: Float) -> String {
        if abs(bearing) < 10 { return "devant vous" }
        if bearing < -45 { return "à gauche" }
        if bearing < 0 { return "légèrement à gauche" }
        if bearing > 45 { return "à droite" }
        return "légèrement à droite"
    }
}


final class UrbanAnalyzer {
    func analyzeMeshes(session: ARSession, camera: ARCamera) -> [UrbanHint] {
        guard let frame = session.currentFrame else { return [] }
        var hints: [UrbanHint] = []

        let camPos = SIMD3<Float>(camera.transform.columns.3.x, camera.transform.columns.3.y, camera.transform.columns.3.z)
        let camForward = normalize(SIMD3<Float>(-camera.transform.columns.2.x, -camera.transform.columns.2.y, -camera.transform.columns.2.z))

        for case let meshAnchor as ARMeshAnchor in frame.anchors {
            let geometry = meshAnchor.geometry
            let vertices = geometry.vertices
            let faces = geometry.faces

            var closest: (dist: Float, pos: SIMD3<Float>, label: String)?

            for faceIdx in 0..<faces.count {
                // ✅ On utilise notre extension propre à ARMeshGeometry
                let classification = geometry.classificationOf(faceWithIndex: faceIdx)
                guard let label = Self.labelFor(classification: classification) else { continue }

                let indices = geometry.indicesOf(faceWithIndex: faceIdx)
                for idx in indices {
                    let vertex = vertices.position(at: idx)
                    let vertexWorld4 = meshAnchor.transform * SIMD4<Float>(vertex.x, vertex.y, vertex.z, 1)
                    let vertexWorld = SIMD3<Float>(vertexWorld4.x, vertexWorld4.y, vertexWorld4.z)
                    let d = simd_distance(vertexWorld, camPos)

                    if d < 6 {
                        if closest == nil || d < closest!.dist {
                            closest = (d, vertexWorld, label)
                        }
                    }
                }
            }

            if let c = closest {
                let bearing = bearingDeg(from: camPos, forward: camForward, to: c.pos)
                hints.append(UrbanHint(label: c.label, distanceMeters: c.dist, bearingDeg: bearing))
            }
        }

        return hints
    }

    private static func labelFor(classification: ARMeshClassification) -> String? {
        switch classification {
        case .wall: return "mur"
        case .door: return "porte"
        case .window: return "fenêtre"
        case .floor: return "sol"
        case .ceiling: return "plafond"
        case .table: return "table"
        case .seat: return "siège"
        default: return nil
        }
    }
}

// MARK: - AR Helpers
private extension ARGeometrySource {
    var count: Int { buffer.length / stride }

    func position(at index: Int) -> SIMD3<Float> {
        let offset = self.offset + index * self.stride
        let ptr = buffer.contents().advanced(by: offset).bindMemory(to: Float.self, capacity: 3)
        return SIMD3<Float>(ptr[0], ptr[1], ptr[2])
    }
}

// ✅ Nouvelle extension ARMeshGeometry corrigée
private extension ARMeshGeometry {
    func classificationOf(faceWithIndex index: Int) -> ARMeshClassification {
        // On récupère la classification si dispo, sinon .none
        guard let classifications = self.classification else { return .none }

        let faces = self.faces
        let indexCountPerPrimitive = 3
        let indexStride = faces.bytesPerIndex
        let basePtr = faces.buffer.contents() // ✅ plus de offset ici

        let indices = (0..<indexCountPerPrimitive).compactMap { v -> Int? in
            let offset = (index * indexCountPerPrimitive + v) * indexStride
            let ptr = basePtr.advanced(by: offset)
            switch indexStride {
            case 2:
                return Int(ptr.bindMemory(to: UInt16.self, capacity: 1).pointee)
            case 4:
                return Int(ptr.bindMemory(to: UInt32.self, capacity: 1).pointee)
            default:
                return nil
            }
        }

        // On prend le premier sommet du triangle pour extraire sa classification
        if let firstIndex = indices.first {
            let clsPtr = classifications.buffer.contents().advanced(by: classifications.offset + firstIndex * classifications.stride)
            let rawU8 = clsPtr.bindMemory(to: UInt8.self, capacity: 1).pointee
            let raw = Int(rawU8)                           // << cast en Int
            return ARMeshClassification(rawValue: raw) ?? .none
        }

        return .none
    }

    func indicesOf(faceWithIndex index: Int) -> [Int] {
        let faces = self.faces
        guard faces.primitiveType == .triangle else { return [] }

        let indexCountPerPrimitive = 3
        let indexStride = faces.bytesPerIndex
        let basePtr = faces.buffer.contents() // ✅ suppression du .offset

        var result: [Int] = []
        for v in 0..<indexCountPerPrimitive {
            let offset = (index * indexCountPerPrimitive + v) * indexStride
            let ptr = basePtr.advanced(by: offset)
            switch indexStride {
            case 2:
                result.append(Int(ptr.bindMemory(to: UInt16.self, capacity: 1).pointee))
            case 4:
                result.append(Int(ptr.bindMemory(to: UInt32.self, capacity: 1).pointee))
            default:
                break
            }
        }
        return result
    }
}

// MARK: - Vision Detector
struct DetectionResult { let label: String; let bbox: CGRect; let confidence: Float }

final class VisionObjectDetector {
    // Replace this with your compiled CoreML model (mlmodelc) name
    // Example: let model = try? VNCoreMLModel(for: YOLOv8n(configuration: MLModelConfiguration()).model)
    private let model: VNCoreMLModel? = {
        do {
            let config = MLModelConfiguration()
            let coreModel = try yolov8n(configuration: config).model
            return try VNCoreMLModel(for: coreModel)
        } catch {
            print("Erreur chargement modèle YOLOv8n: \(error)")
            return nil
        }
    }()

    private let ciContext = CIContext()

    func detectObjects(on frame: ARFrame, in arView: ARView) -> [UrbanHint] {
        guard let mlModel = model else { return [] }
        let handler = VNImageRequestHandler(cvPixelBuffer: frame.capturedImage, orientation: .up, options: [:])
        var results: [DetectionResult] = []
        let request = VNCoreMLRequest(model: mlModel) { req, _ in
            for case let obs as VNRecognizedObjectObservation in (req.results ?? []) {
                guard let top = obs.labels.first else { continue }
                results.append(DetectionResult(label: top.identifier, bbox: obs.boundingBox, confidence: top.confidence))
            }
        }
        // Small input-side crop/scale handled by Vision
        try? handler.perform([request])

        // Map each detection's 2D center to a 3D point via LiDAR raycast
        var hints: [UrbanHint] = []
        for det in results where det.confidence >= 0.5 {
            let screenPoint = visionBBoxCenterToScreenPoint(det.bbox, in: arView)
            let hits = arView.raycast(from: screenPoint, allowing: .estimatedPlane, alignment: .any)
            if let h = hits.first, let cam = arView.session.currentFrame?.camera {
                let camPos = SIMD3<Float>(cam.transform.columns.3.x, cam.transform.columns.3.y, cam.transform.columns.3.z)
                let hp = SIMD3<Float>(h.worldTransform.columns.3.x, h.worldTransform.columns.3.y, h.worldTransform.columns.3.z)
                let dist = simd_distance(camPos, hp)
                if dist < 10 {
                    let forward = normalize(SIMD3<Float>(-cam.transform.columns.2.x, -cam.transform.columns.2.y, -cam.transform.columns.2.z))
                    let bearing = bearingDeg(from: camPos, forward: forward, to: hp)
                    // Map common classes to French labels; default to identifier
                    let label = frenchLabel(for: det.label)
                    hints.append(UrbanHint(label: label, distanceMeters: dist, bearingDeg: bearing))
                }
            }
        }
        return hints
    }

    private func frenchLabel(for id: String) -> String {
        let map: [String: String] = [
            "person": "personne",
            "bicycle": "vélo",
            "car": "voiture",
            "motorcycle": "moto",
            "bus": "bus",
            "truck": "camion",
            "traffic light": "feu tricolore",
            "stop sign": "panneau stop",
            "bench": "banc",
            "crosswalk": "passage piéton",
            "pole": "poteau"
        ]
        return map[id, default: id]
    }

    private func visionBBoxCenterToScreenPoint(_ bbox: CGRect, in arView: ARView) -> CGPoint {
        // Vision bbox is normalized in [0,1], origin bottom-left; UIKit is top-left
        let nx = bbox.midX
        let ny = 1.0 - bbox.midY
        return CGPoint(x: nx * arView.bounds.width, y: ny * arView.bounds.height)
    }
}

// MARK: - Narration
final class DescriptionSpeaker {
    private let synth = AVSpeechSynthesizer()
    private let voice = AVSpeechSynthesisVoice(language: "fr-FR")
    func speak(_ text: String) {
        let utt = AVSpeechUtterance(string: text)
        utt.voice = voice
        utt.rate = AVSpeechUtteranceDefaultSpeechRate * 0.9
        synth.speak(utt)
    }
}

// MARK: - Math helpers
func bearingDeg(from cameraPos: SIMD3<Float>, forward: SIMD3<Float>, to worldPos: SIMD3<Float>) -> Float {
    let toVec = normalize(worldPos - cameraPos)
    // Project onto horizontal plane to avoid pitch influencing sign
    let f = normalize(SIMD3<Float>(forward.x, 0, forward.z))
    let t = normalize(SIMD3<Float>(toVec.x, 0, toVec.z))
    let dotv = max(-1.0 as Float, min(1.0 as Float, simd_dot(f, t)))
    let angle = acos(dotv) * 180 / .pi
    // Determine sign via cross product (positive = right)
    let crossY = (f.x * t.z) - (f.z * t.x)
    return crossY >= 0 ? angle : -angle
}
