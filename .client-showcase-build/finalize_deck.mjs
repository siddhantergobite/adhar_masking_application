import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const workspaceDir = process.cwd();
const buildDir = path.join(workspaceDir, ".client-showcase-build");
const skillDir = "C:\\Users\\admin\\.codex\\plugins\\cache\\openai-primary-runtime\\presentations\\26.1017.22136\\skills\\presentations";
const candidatePath = path.join(buildDir, "Aadhaar_Masking_Equitas_Client_Showcase.draft.pptx");
const finalPath = path.join(workspaceDir, "output", "Aadhaar_Masking_Equitas_Client_Showcase_finalized_v2.pptx");
const expectedSlideSizeEmu = "12192000,6858000";
const requirements = {
  explicitTotalSlideCount: 5,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
};

const { finalizePresentation } = await import(pathToFileURL(
  path.join(skillDir, "container_tools", "artifact_tool_utils.mjs"),
).href);
const stagingDir = path.join(buildDir, "finalizer-staging");
await fs.mkdir(stagingDir, { recursive: true });
await fs.mkdir(path.dirname(finalPath), { recursive: true });

const result = await finalizePresentation({
  ...requirements,
  workspaceDir,
  candidatePath,
  finalPath,
  pythonExecutable: "C:\\Users\\admin\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe",
  integrityValidatorPath: path.join(skillDir, "container_tools", "inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools", "inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", expectedSlideSizeEmu,
    "--validate-bullet-geometry",
    "--validate-heading-fit",
  ],
  requiredNativeTableOwnerSlides: [],
  fontPolicy: { basis: "design", families: ["Arial"] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "Aadhaar_Masking_Equitas_Client_Showcase_finalized_v2.validation.json"),
});

console.log(JSON.stringify(result, null, 2));
