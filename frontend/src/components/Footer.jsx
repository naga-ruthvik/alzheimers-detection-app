import { Brain, Github, Twitter, Linkedin, Heart } from 'lucide-react';

export default function Footer() {
    return (
        <footer className="w-full bg-gray-900 border-t border-gray-800 pt-16 mt-16 pb-8 relative z-20">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="grid grid-cols-1 md:grid-cols-4 gap-12 md:gap-8 mb-12">

                    {/* Brand Info */}
                    <div className="col-span-1 md:col-span-1 space-y-4 text-center md:text-left">
                        <div className="flex items-center justify-center md:justify-start space-x-3 mb-4">
                            <Brain className="w-6 h-6 text-brand-400" />
                            <span className="font-bold text-xl text-white">XAI-TransMed</span>
                        </div>
                        <p className="text-sm text-gray-400 leading-relaxed max-w-sm mx-auto md:mx-0">
                            Transforming Alzheimer's diagnosis with state-of-the-art interpretable deep learning. Providing transparency to clinical AI pipelines.
                        </p>
                    </div>

                    {/* Quick Links */}
                    <div className="flex flex-col items-center md:items-start text-sm">
                        <h4 className="font-semibold text-white uppercase tracking-wider mb-4">Project</h4>
                        <div className="space-y-3 flex flex-col items-center md:items-start">
                            <a href="#" className="text-gray-400 hover:text-brand-400 transition-colors">Architecture Overview</a>
                            <a href="#" className="text-gray-400 hover:text-brand-400 transition-colors">Model Explainability</a>
                            <a href="#" className="text-gray-400 hover:text-brand-400 transition-colors">Dataset Information</a>
                            <a href="#" className="text-gray-400 hover:text-brand-400 transition-colors">API Documentation</a>
                        </div>
                    </div>

                    {/* Legal / Ethics */}
                    <div className="flex flex-col items-center md:items-start text-sm">
                        <h4 className="font-semibold text-white uppercase tracking-wider mb-4">Clinical</h4>
                        <div className="space-y-3 flex flex-col items-center md:items-start">
                            <a href="#" className="text-gray-400 hover:text-brand-400 transition-colors">Research Ethics</a>
                            <a href="#" className="text-gray-400 hover:text-brand-400 transition-colors">Data Privacy Status</a>
                            <a href="#" className="text-gray-400 hover:text-brand-400 transition-colors">Limitations</a>
                            <a href="#" className="text-gray-400 hover:text-brand-400 transition-colors">Medical Disclaimer</a>
                        </div>
                    </div>

                    {/* Social / Connect */}
                    <div className="flex flex-col items-center md:items-start">
                        <h4 className="font-semibold text-white uppercase tracking-wider mb-4 text-sm">Connect</h4>
                        <div className="flex space-x-4">
                            <a href="#" className="p-2 bg-gray-800 rounded-full text-gray-400 hover:text-white hover:bg-brand-600 transition-all">
                                <Github className="w-5 h-5" />
                            </a>
                            <a href="#" className="p-2 bg-gray-800 rounded-full text-gray-400 hover:text-white hover:bg-brand-500 transition-all">
                                <Twitter className="w-5 h-5" />
                            </a>
                            <a href="#" className="p-2 bg-gray-800 rounded-full text-gray-400 hover:text-white hover:bg-brand-600 transition-all">
                                <Linkedin className="w-5 h-5" />
                            </a>
                        </div>
                    </div>

                </div>

                {/* Bottom Bar */}
                <div className="border-t border-gray-800 pt-8 flex flex-col md:flex-row items-center justify-between">
                    <p className="text-xs text-gray-500 flex items-center mb-4 md:mb-0">
                        Made with <Heart className="w-3 h-3 text-red-500 mx-1" /> by XAI-TransMed Research Team © {new Date().getFullYear()}
                    </p>
                    <div className="text-xs font-mono text-gray-500/50 bg-gray-800/50 px-3 py-1 rounded-full border border-gray-700/50">
                        PRE-CLINICAL ALPHA BUILD v0.9.2
                    </div>
                </div>
            </div>
        </footer>
    );
}
