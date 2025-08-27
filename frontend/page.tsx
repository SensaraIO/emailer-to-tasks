import Link from 'next/link'

export default function HomePage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="max-w-md w-full space-y-8">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Gmail to Task Board
          </h1>
          <p className="text-gray-600 mb-8">
            Convert your emails into actionable tasks using AI
          </p>
          <div className="space-y-4">
            <Link
              href="/tasks"
              className="block w-full text-center bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition"
            >
              View Task Board
            </Link>
            <a
              href={`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080'}/oauth/google/start`}
              className="block w-full text-center bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700 transition"
            >
              Connect Gmail Account
            </a>
          </div>
        </div>
      </div>
    </div>
  )
}